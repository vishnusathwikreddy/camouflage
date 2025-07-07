"""Data class, holding information about dataloaders and poison ids."""

import torch
import numpy as np

import pickle

import datetime
import os
import warnings
import random
import PIL

from .datasets import construct_datasets, Subset
from .cached_dataset import CachedDataset

from .diff_data_augmentation import RandomTransform

from ..consts import PIN_MEMORY, BENCHMARK, DISTRIBUTED_BACKEND, SHARING_STRATEGY, MAX_THREADING
from ..utils import set_random_seed
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Kettle():
    """Brew poison with given arguments.

    Data class.
    Attributes:
    - trainloader
    - validloader
    - poisonloader
    - poison_ids
    - trainset/poisonset/targetset

    Most notably .poison_lookup is a dictionary that maps image ids to their slice in the poison_delta tensor.

    Initializing this class will set up all necessary attributes.

    Other data-related methods of this class:
    - initialize_poison
    - export_poison

    """

    def __init__(self, args, batch_size, augmentations, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with given specs..."""
        self.args, self.setup = args, setup
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.trainset, self.validset = self.prepare_data(normalize=True)
        num_workers = self.get_num_workers()

        if self.args.lmdb_path is not None:
            from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb
            self.trainset = LMDBDataset(self.trainset, self.args.lmdb_path, 'train')
            self.validset = LMDBDataset(self.validset, self.args.lmdb_path, 'val')

        if self.args.cache_dataset:
            self.trainset = CachedDataset(self.trainset, num_workers=num_workers)
            self.validset = CachedDataset(self.validset, num_workers=num_workers)
            num_workers = 0

        if self.args.poisonkey is None:
            if self.args.benchmark != '':
                with open(self.args.benchmark, 'rb') as handle:
                    setup_dict = pickle.load(handle)
                self.benchmark_construction(setup_dict[self.args.benchmark_idx])  # using the first setup dict for benchmarking
            else:
                self.random_construction()


        else:
            if '-' in self.args.poisonkey:
                # If the poisonkey contains a dash-separated triplet like 5-3-1, then poisons are drawn
                # entirely deterministically.
                self.deterministic_construction()
            else:
                # Otherwise the poisoning process is random.
                # If the poisonkey is a random integer, then this integer will be used
                # as a key to seed the random generators.
                self.random_construction()


        # Generate loaders:
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=min(self.batch_size, len(self.trainset)),
                                                       shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.validloader = torch.utils.data.DataLoader(self.validset, batch_size=min(self.batch_size, len(self.validset)),
                                                       shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        validated_batch_size = max(min(args.pbatch, len(self.poisonset)), 1)
        self.poisonloader = torch.utils.data.DataLoader(self.poisonset, batch_size=validated_batch_size,
                                                        shuffle=self.args.pshuffle, drop_last=False, num_workers=num_workers,
                                                        pin_memory=PIN_MEMORY)

        # Initialize camouflage data structures (will be set up later if needed)
        self.camouflageset = None
        self.camouflage_ids = None
        self.camouflage_lookup = None
        self.camouflageloader = None

        # Ablation on a subset?
        if args.ablation < 1.0:
            self.sample = random.sample(range(len(self.trainset)), int(self.args.ablation * len(self.trainset)))
            self.partialset = Subset(self.trainset, self.sample)
            self.partialloader = torch.utils.data.DataLoader(self.partialset, batch_size=min(self.batch_size, len(self.partialset)),
                                                             shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
        self.print_status()


    """ STATUS METHODS """

    def print_status(self):
        class_names = self.trainset.classes
        print(
            f'Poisoning setup generated for threat model {self.args.threatmodel} and '
            f'budget of {self.args.budget * 100}% - {len(self.poisonset)} images:')
        print(
            f'--Target images drawn from class {", ".join([class_names[self.targetset[i][1]] for i in range(len(self.targetset))])}'
            f' with ids {self.target_ids}.')
        print(f'--Target images assigned intended class {", ".join([class_names[i] for i in self.poison_setup["intended_class"]])}.')

        if self.poison_setup["poison_class"] is not None:
            print(f'--Poison images drawn from class {class_names[self.poison_setup["poison_class"]]}.')
        else:
            print(f'--Poison images drawn from all classes.')

        if self.args.ablation < 1.0:
            print(f'--Partialset is {len(self.partialset)/len(self.trainset):2.2%} of full training set')
            num_p_poisons = len(np.intersect1d(self.poison_ids.cpu().numpy(), np.array(self.sample)))
            print(f'--Poisons in partialset are {num_p_poisons} ({num_p_poisons/len(self.poison_ids):2.2%})')

    def get_num_workers(self):
        """Check devices and set an appropriate number of workers."""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            max_num_workers = 4 * num_gpus
        else:
            max_num_workers = 4
        if torch.get_num_threads() > 1 and MAX_THREADING > 0:
            worker_count = min(min(2 * torch.get_num_threads(), max_num_workers), MAX_THREADING)
        else:
            worker_count = 0
        # worker_count = 200
        print(f'Data is loaded with {worker_count} workers.')
        return worker_count

    """ CONSTRUCTION METHODS """

    def prepare_data(self, normalize=True):
        trainset, validset = construct_datasets(self.args.dataset, self.args.data_path, normalize)


        # Prepare data mean and std for later:
        self.dm = torch.tensor(trainset.data_mean)[None, :, None, None].to(**self.setup)
        self.ds = torch.tensor(trainset.data_std)[None, :, None, None].to(**self.setup)


        # Train augmentations are handled separately as they possibly have to be backpropagated
        if self.augmentations is not None or self.args.paugment:
            if 'CIFAR' in self.args.dataset:
                params = dict(source_size=32, target_size=32, shift=8, fliplr=True)
            elif 'MNIST' in self.args.dataset:
                params = dict(source_size=28, target_size=28, shift=4, fliplr=True)
            elif 'TinyImageNet' in self.args.dataset:
                params = dict(source_size=64, target_size=64, shift=64 // 4, fliplr=True)
            elif 'ImageNet' in self.args.dataset:
                params = dict(source_size=224, target_size=224, shift=224 // 4, fliplr=True)

            if self.augmentations == 'default':
                self.augment = RandomTransform(**params, mode='bilinear')
            elif not self.defs.augmentations:
                print('Data augmentations are disabled.')
                self.augment = RandomTransform(**params, mode='bilinear')
            else:
                raise ValueError(f'Invalid diff. transformation given: {self.augmentations}.')

        return trainset, validset

    def deterministic_construction(self):
        """Construct according to the triplet input key.

        The triplet key, e.g. 5-3-1 denotes in order:
        target_class - poison_class - target_id

        Poisons are always the first n occurences of the given class.
        [This is the same setup as in metapoison]
        """
        if self.args.threatmodel != 'single-class':
            raise NotImplementedError()

        split = self.args.poisonkey.split('-')
        if len(split) != 3:
            raise ValueError('Invalid poison triplet supplied.')
        else:
            target_class, poison_class, target_id = [int(s) for s in split]
        self.init_seed = self.args.poisonkey
        print(f'Initializing Poison data (chosen images, examples, targets, labels) as {self.args.poisonkey}')

        self.poison_setup = dict(poison_budget=self.args.budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.poisonset, self.targetset, self.validset = self._choose_poisons_deterministic(target_id)

    def benchmark_construction(self, setup_dict):
        """Construct according to the benchmark."""
        target_class, poison_class = setup_dict['target class'], setup_dict['base class']

        budget = len(setup_dict['base indices']) / len(self.trainset)
        self.poison_setup = dict(poison_budget=budget,
                                 target_num=self.args.targets, poison_class=poison_class, target_class=target_class,
                                 intended_class=[poison_class])
        self.init_seed = self.args.poisonkey
        self.poisonset, self.targetset, self.validset = self._choose_poisons_benchmark(setup_dict)

    def _choose_poisons_benchmark(self, setup_dict):
        # poisons
        class_ids = setup_dict['base indices']
        poison_num = len(class_ids)
        self.poison_ids = class_ids

        # the target
        self.target_ids = [setup_dict['target index']]
        # self.target_ids = setup_dict['target index']

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))

        return poisonset, targetset, validset

    def _choose_poisons_deterministic(self, target_id):
        # poisons
        class_ids = []
        for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
            target, idx = self.trainset.get_target(index)
            if target == self.poison_setup['poison_class']:
                class_ids.append(idx)

        poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
        if len(class_ids) < poison_num:
            warnings.warn(f'Training set is too small for requested poison budget.')
            poison_num = len(class_ids)
        self.poison_ids = class_ids[:poison_num]

        # the target
        # class_ids = []
        # for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
        #     target, idx = self.validset.get_target(index)
        #     if target == self.poison_setup['target_class']:
        #         class_ids.append(idx)
        # self.target_ids = [class_ids[target_id]]
        # Disable for now for benchmark sanity check. This is a breaking change.
        self.target_ids = [target_id]

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids, range(poison_num)))
        dict(zip(self.poison_ids, range(poison_num)))
        return poisonset, targetset, validset

    def random_construction(self):
        """Construct according to random selection.

        The setup can be repeated from its key (which initializes the random generator).
        This method sets
         - poison_setup
         - poisonset / targetset / validset

        """
        if self.args.local_rank is None:
            if self.args.poisonkey is None:
                self.init_seed = np.random.randint(0, 2**32 - 1)
            else:
                self.init_seed = int(self.args.poisonkey)
            set_random_seed(self.init_seed)
            print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {self.init_seed}')
        else:
            rank = torch.distributed.get_rank()
            if self.args.poisonkey is None:
                init_seed = torch.randint(0, 2**32 - 1, [1], device=self.setup['device'])
            else:
                init_seed = torch.as_tensor(int(self.args.poisonkey), dtype=torch.int64, device=self.setup['device'])
            torch.distributed.broadcast(init_seed, src=0)
            if rank == 0:
                print(f'Initializing Poison data (chosen images, examples, targets, labels) with random seed {init_seed.item()}')
            self.init_seed = init_seed.item()
            set_random_seed(self.init_seed)
        # Parse threat model
        self.poison_setup = self._parse_threats_randomly()
        self.poisonset, self.targetset, self.validset = self._choose_poisons_randomly()

    def _parse_threats_randomly(self):
        """Parse the different threat models.

        The threat-models are [In order of expected difficulty]:

        single-class replicates the threat model of feature collision attacks,
        third-party draws all poisons from a class that is unrelated to both target and intended label.
        random-subset draws poison images from all classes.
        random-subset draw poison images from all classes and draws targets from different classes to which it assigns
        different labels.
        """
        num_classes = len(self.trainset.classes)

        target_class = np.random.randint(num_classes)
        list_intentions = list(range(num_classes))
        list_intentions.remove(target_class)
        intended_class = [np.random.choice(list_intentions)] * self.args.targets

        if self.args.targets < 1:
            poison_setup = dict(poison_budget=0, target_num=0,
                                poison_class=np.random.randint(num_classes), target_class=None,
                                intended_class=[np.random.randint(num_classes)])
            warnings.warn('Number of targets set to 0.')
            return poison_setup

        if self.args.threatmodel == 'single-class':
            poison_class = intended_class[0]
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'third-party':
            list_intentions.remove(intended_class[0])
            poison_class = np.random.choice(list_intentions)
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'self-betrayal':
            poison_class = target_class
            poison_setup = dict(poison_budget=self.args.budget, target_num=self.args.targets,
                                poison_class=poison_class, target_class=target_class, intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset':
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=target_class,
                                intended_class=intended_class)
        elif self.args.threatmodel == 'random-subset-random-targets':
            target_class = None
            intended_class = np.random.randint(num_classes, size=self.args.targets)
            poison_class = None
            poison_setup = dict(poison_budget=self.args.budget,
                                target_num=self.args.targets, poison_class=None, target_class=None,
                                intended_class=intended_class)
        else:
            raise NotImplementedError('Unknown threat model.')

        return poison_setup

    def _choose_poisons_randomly(self):
        """Subconstruct poison and targets.

        The behavior is different for poisons and targets. We still consider poisons to be part of the original training
        set and load them via trainloader (And then add the adversarial pattern Delta)
        The targets are fully removed from the validation set and returned as a separate dataset, indicating that they
        should not be considered during clean validation using the validloader

        """
        # Poisons:
        if self.poison_setup['poison_class'] is not None:
            class_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.trainset.get_target(index)
                if target == self.poison_setup['poison_class']:
                    class_ids.append(idx)

            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(class_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(class_ids)}')
                poison_num = len(class_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                class_ids, size=poison_num, replace=False), dtype=torch.long)
        else:
            total_ids = []
            for index in range(len(self.trainset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.trainset.get_target(index)
                total_ids.append(idx)
            poison_num = int(np.ceil(self.args.budget * len(self.trainset)))
            if len(total_ids) < poison_num:
                warnings.warn(f'Training set is too small for requested poison budget. \n'
                              f'Budget will be reduced to maximal size {len(total_ids)}')
                poison_num = len(total_ids)
            self.poison_ids = torch.tensor(np.random.choice(
                total_ids, size=poison_num, replace=False), dtype=torch.long)

        # Targets:
        if self.poison_setup['target_class'] is not None:
            class_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                target, idx = self.validset.get_target(index)
                if target == self.poison_setup['target_class']:
                    class_ids.append(idx)
            self.target_ids = np.random.choice(class_ids, size=self.args.targets, replace=False)
        else:
            total_ids = []
            for index in range(len(self.validset)):  # we actually iterate this way not to iterate over the images
                _, idx = self.validset.get_target(index)
                total_ids.append(idx)
            self.target_ids = np.random.choice(total_ids, size=self.args.targets, replace=False)

        targetset = Subset(self.validset, indices=self.target_ids)
        valid_indices = []
        for index in range(len(self.validset)):
            _, idx = self.validset.get_target(index)
            if idx not in self.target_ids:
                valid_indices.append(idx)
        validset = Subset(self.validset, indices=valid_indices)
        poisonset = Subset(self.trainset, indices=self.poison_ids)

        # Construct lookup table
        self.poison_lookup = dict(zip(self.poison_ids.tolist(), range(poison_num)))
        return poisonset, targetset, validset

    def initialize_poison(self, initializer=None):
        """Initialize according to args.init.

        Propagate initialization in distributed settings.
        """
        if initializer is None:
            initializer = self.args.init

        # ds has to be placed on the default (cpu) device, not like self.ds
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        if initializer == 'zero':
            init = torch.zeros(len(self.poison_ids), *self.trainset[0][0].shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.poison_ids), *self.trainset[0][0].shape) - 0.5) * 2
            init *= self.args.eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
            init *= self.args.eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.poison_ids), *self.trainset[0][0].shape)
        else:
            raise NotImplementedError()

        init.data = torch.max(torch.min(init, self.args.eps / ds / 255), -self.args.eps / ds / 255)

        # If distributed, sync poison initializations
        if self.args.local_rank is not None:
            if DISTRIBUTED_BACKEND == 'nccl':
                init = init.to(device=self.setup['device'])
                torch.distributed.broadcast(init, src=0)
                init.to(device=torch.device('cpu'))
            else:
                torch.distributed.broadcast(init, src=0)
        return init

    """ EXPORT METHODS """

    def export_poison(self, poison_delta, path=None, mode='automl'):
        """Export poisons in either packed mode (just ids and raw data) or in full export mode, exporting all images.

        In full export mode, export data into folder structure that can be read by a torchvision.datasets.ImageFolder

        In automl export mode, export data into a single folder and produce a csv file that can be uploaded to
        google storage.
        """
        if path is None:
            path = self.args.poison_path

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.poison_lookup.get(idx)
            if (lookup is not None) and train:
                input += poison_delta[lookup, :, :, :]
            _torch_to_PIL(input).save(filename)

        # Save either into packed mode, ImageDataSet Mode or google storage mode
        if mode == 'packed':
            # Ensure directory exists
            os.makedirs(path, exist_ok=True)
            
            data = dict()
            data['poison_setup'] = self.poison_setup
            data['poison_delta'] = poison_delta
            data['poison_ids'] = self.poison_ids
            data['target_images'] = [target for target in self.targetset]
            name = f'poisons_packed_{datetime.date.today()}.pth'
            
            # Debug information
            print(f'DEBUG: Saving poison data to {os.path.join(path, name)}')
            print(f'DEBUG: poison_delta shape: {poison_delta.shape if poison_delta is not None else "None"}')
            print(f'DEBUG: poison_ids length: {len(self.poison_ids) if self.poison_ids is not None else "None"}')
            print(f'DEBUG: data dict keys: {list(data.keys())}')
            
            torch.save(data, os.path.join(path, name))
            print(f'DEBUG: torch.save completed for {name}')
            
            # Verify the file was created and has content
            file_path = os.path.join(path, name)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f'DEBUG: File created successfully: {file_path}, size: {file_size} bytes')
            else:
                print(f'ERROR: File was not created: {file_path}')

        elif mode == 'limited':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Poisoned training images exported ...')

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')

        elif mode in ['automl-upload', 'automl-all', 'automl-baseline']:
            from ..utils import automl_bridge
            targetclass = self.targetset[0][1]
            poisonclass = self.poison_setup["poison_class"]

            name_candidate = f'{self.args.name}_{self.args.dataset}T{targetclass}P{poisonclass}'
            name = ''.join(e for e in name_candidate if e.isalnum())

            if mode == 'automl-upload':
                automl_phase = 'poison-upload'
            elif mode == 'automl-all':
                automl_phase = 'all'
            elif mode == 'automl-baseline':
                automl_phase = 'upload'
            automl_bridge(self, poison_delta, name, mode=automl_phase, dryrun=self.args.dryrun)

        elif mode == 'numpy':
            _, h, w = self.trainset[0][0].shape
            training_data = np.zeros([len(self.trainset), h, w, 3])
            labels = np.zeros(len(self.trainset))
            for input, label, idx in self.trainset:
                lookup = self.poison_lookup.get(idx)
                if lookup is not None:
                    input += poison_delta[lookup, :, :, :]
                training_data[idx] = np.asarray(_torch_to_PIL(input))
                labels[idx] = label

            np.save(os.path.join(path, 'poisoned_training_data.npy'), training_data)
            np.save(os.path.join(path, 'poisoned_training_labels.npy'), labels)

        elif mode == 'kettle-export':
            with open(f'kette_{self.args.dataset}{self.args.model}.pkl', 'wb') as file:
                pickle.dump([self, poison_delta], file, protocol=pickle.HIGHEST_PROTOCOL)

        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx), "poisons") # save poisons and camouflages in different folders
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_poisons = []
            for lookup, key in enumerate(self.poison_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.trainset[key]
                input += poison_delta[lookup, :, :, :]
                benchmark_poisons.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'poisons.pickle'), 'wb+') as file:
                pickle.dump(benchmark_poisons, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Target
            target, target_label, _ = self.targetset[0]
            with open(os.path.join(sub_path, 'target.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(target), target_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.poison_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            raise NotImplementedError()

        print('Dataset fully exported.')

    """ CAMOUFLAGE METHODS """

    def setup_camouflage(self, camouflage_budget=None, camouflage_key=None):
        """Set up camouflage data structures for defense against poison detection.
        
        Camouflage samples are drawn from clean images with the same label as targets.
        They are optimized to align their gradients with target gradients using a 
        model trained on clean + poison data.
        
        Args:
            camouflage_budget: Fraction of training data to use for camouflage
            camouflage_key: Random seed for reproducible camouflage selection
        """
        if camouflage_budget is None:
            camouflage_budget = getattr(self.args, 'camouflage_budget', 0.01)
        if camouflage_key is None:
            camouflage_key = getattr(self.args, 'camouflage_key', None)
            
        # Set random seed for reproducible camouflage selection
        if camouflage_key is not None:
            set_random_seed(camouflage_key)
            print(f'Initializing camouflage data with random seed {camouflage_key}')
        else:
            # Use a different seed than poison if not specified
            camouflage_seed = self.init_seed + 12345 if hasattr(self, 'init_seed') else np.random.randint(0, 2**32 - 1)
            set_random_seed(camouflage_seed)
            print(f'Initializing camouflage data with random seed {camouflage_seed}')
        
        # Get target labels - camouflage samples must have the same label as targets
        target_labels = [data[1] for data in self.targetset]
        print(f'Target labels for camouflage: {target_labels}')
        
        # Find all training samples with target labels (excluding poison samples)
        candidate_indices = []
        for idx in range(len(self.trainset)):
            label, _ = self.trainset.get_target(idx)
            # Include if: has target label AND not already a poison sample
            if label in target_labels and idx not in self.poison_ids:
                candidate_indices.append(idx)
        
        print(f'Found {len(candidate_indices)} candidate images with target labels {target_labels} for camouflage')
        print(f'Training set size: {len(self.trainset)}, Poison IDs: {len(self.poison_ids)}')
        
        # Sample camouflage images from candidates
        camouflage_num = int(camouflage_budget * len(self.trainset))
        print(f'Requested camouflage samples: {camouflage_num} (budget: {camouflage_budget})')
        camouflage_num = min(camouflage_num, len(candidate_indices))  # Don't exceed available candidates
        
        if camouflage_num == 0:
            print('Warning: No camouflage samples can be created. Insufficient candidates.')
            print(f'DEBUG: camouflage_budget={camouflage_budget}, trainset_size={len(self.trainset)}')
            print(f'DEBUG: candidate_indices={len(candidate_indices)}, target_labels={target_labels}')
            self.camouflage_ids = torch.tensor([])
            self.camouflageset = Subset(self.trainset, indices=[])
            self.camouflage_lookup = {}
            return
            
        self.camouflage_ids = torch.tensor(random.sample(candidate_indices, camouflage_num))
        print(f'Selected {len(self.camouflage_ids)} images for camouflage generation')
        
        # Create camouflage dataset and lookup
        self.camouflageset = Subset(self.trainset, indices=self.camouflage_ids)
        self.camouflage_lookup = dict(zip(self.camouflage_ids.tolist(), range(len(self.camouflage_ids))))
        
        # Create camouflage loader
        num_workers = self.get_num_workers()
        validated_batch_size = max(min(self.args.pbatch, len(self.camouflageset)), 1)
        self.camouflageloader = torch.utils.data.DataLoader(
            self.camouflageset, 
            batch_size=validated_batch_size,
            shuffle=self.args.pshuffle, 
            drop_last=False, 
            num_workers=num_workers,
            pin_memory=PIN_MEMORY
        )
        
        print(f'Camouflage setup complete: {len(self.camouflage_ids)} samples, batch size {validated_batch_size}')

    def initialize_camouflage(self, initializer=None):
        """Initialize camouflage perturbations.
        
        Args:
            initializer: Initialization method ('zero', 'rand', 'randn', 'normal')
            
        Returns:
            torch.Tensor: Initial camouflage perturbations
        """
        if self.camouflage_ids is None or len(self.camouflage_ids) == 0:
            print('Warning: No camouflage IDs set. Call setup_camouflage() first.')
            return torch.zeros(0, *self.trainset[0][0].shape)
            
        if initializer is None:
            initializer = getattr(self.args, 'camouflage_init', self.args.init)

        # Use same initialization logic as poison
        ds = torch.tensor(self.trainset.data_std)[None, :, None, None]
        eps = getattr(self.args, 'camouflage_eps', self.args.eps)
        
        if initializer == 'zero':
            init = torch.zeros(len(self.camouflage_ids), *self.trainset[0][0].shape)
        elif initializer == 'rand':
            init = (torch.rand(len(self.camouflage_ids), *self.trainset[0][0].shape) - 0.5) * 2
            init *= eps / ds / 255
        elif initializer == 'randn':
            init = torch.randn(len(self.camouflage_ids), *self.trainset[0][0].shape)
            init *= eps / ds / 255
        elif initializer == 'normal':
            init = torch.randn(len(self.camouflage_ids), *self.trainset[0][0].shape)
        else:
            raise NotImplementedError(f'Camouflage initializer {initializer} not implemented')

        # Apply epsilon constraints
        init.data = torch.max(torch.min(init, eps / ds / 255), -eps / ds / 255)

        # If distributed, sync camouflage initializations
        if self.args.local_rank is not None:
            if DISTRIBUTED_BACKEND == 'nccl':
                init = init.to(device=self.setup['device'])
                torch.distributed.broadcast(init, src=0)
                init.to(device=torch.device('cpu'))
            else:
                torch.distributed.broadcast(init, src=0)
                
        print(f'Initialized {len(self.camouflage_ids)} camouflage perturbations with {initializer}')
        return init

    def export_camouflage(self, camouflage_delta, path=None, mode='packed'):
        """Export camouflage samples.
        
        Args:
            camouflage_delta: Camouflage perturbations tensor
            path: Export path
            mode: Export mode ('packed', 'limited', 'full', 'automl')
        """
        if path is None:
            path = self.args.poison_path

        if self.camouflage_ids is None or len(self.camouflage_ids) == 0:
            print('Warning: No camouflage samples to export')
            return

        dm = torch.tensor(self.trainset.data_mean)[:, None, None]
        ds = torch.tensor(self.trainset.data_std)[:, None, None]

        def _torch_to_PIL(image_tensor):
            """Torch->PIL pipeline as in torchvision.utils.save_image."""
            image_denormalized = torch.clamp(image_tensor * ds + dm, 0, 1)
            image_torch_uint8 = image_denormalized.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
            image_PIL = PIL.Image.fromarray(image_torch_uint8.numpy())
            return image_PIL

        def _save_image(input, label, idx, location, train=True):
            """Save input image to given location, add poison_delta if necessary."""
            filename = os.path.join(location, str(idx) + '.png')

            lookup = self.camouflage_lookup.get(idx)
            if (lookup is not None) and train:
                input += camouflage_delta[lookup, :, :, :]
            _torch_to_PIL(input).save(filename)

        if mode == 'packed':
            # Ensure directory exists
            os.makedirs(path, exist_ok=True)
            
            data = dict()
            data['camouflage_setup'] = {
                'camouflage_ids': self.camouflage_ids,
                'camouflage_lookup': self.camouflage_lookup,
                'target_labels': [target[1] for target in self.targetset]
            }
            data['camouflage_delta'] = camouflage_delta
            data['camouflage_ids'] = self.camouflage_ids
            data['target_images'] = [target for target in self.targetset]
            
            filename = f'camouflages_packed_{datetime.date.today()}.pth'
            torch.save(data, os.path.join(path, filename))
            print(f'Camouflage exported in packed mode to {os.path.join(path, filename)}')
        
        elif mode == 'full':
            # Save training set
            names = self.trainset.classes
            for name in names:
                os.makedirs(os.path.join(path, 'train', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'test', name), exist_ok=True)
                os.makedirs(os.path.join(path, 'targets', name), exist_ok=True)
            for input, label, idx in self.trainset:
                _save_image(input, label, idx, location=os.path.join(path, 'train', names[label]), train=True)
            print('Camouflage training images exported ...')

            for input, label, idx in self.validset:
                _save_image(input, label, idx, location=os.path.join(path, 'test', names[label]), train=False)
            print('Unaffected validation images exported ...')

            # Save secret targets
            for enum, (target, _, idx) in enumerate(self.targetset):
                intended_class = self.poison_setup['intended_class'][enum]
                _save_image(target, intended_class, idx, location=os.path.join(path, 'targets', names[intended_class]), train=False)
            print('Target images exported with intended class labels ...')


        elif mode == 'benchmark':
            foldername = f'{self.args.name}_{"_".join(self.args.net)}'
            sub_path = os.path.join(path, 'benchmark_results', foldername, str(self.args.benchmark_idx), "camouflages")  # save poisons and camouflages in different folders
            os.makedirs(sub_path, exist_ok=True)

            # Poisons
            benchmark_camouflages = []
            for lookup, key in enumerate(self.camouflage_lookup.keys()):  # This is a different order than we usually do for compatibility with the benchmark
                input, label, _ = self.trainset[key]
                input += camouflage_delta[lookup, :, :, :]
                benchmark_camouflages.append((_torch_to_PIL(input), int(label)))

            with open(os.path.join(sub_path, 'camouflages.pickle'), 'wb+') as file:
                pickle.dump(benchmark_camouflages, file, protocol=pickle.HIGHEST_PROTOCOL)

            # Target
            target, target_label, _ = self.targetset[0]
            with open(os.path.join(sub_path, 'target.pickle'), 'wb+') as file:
                pickle.dump((_torch_to_PIL(target), target_label), file, protocol=pickle.HIGHEST_PROTOCOL)

            # Indices
            with open(os.path.join(sub_path, 'base_indices.pickle'), 'wb+') as file:
                pickle.dump(self.camouflage_ids, file, protocol=pickle.HIGHEST_PROTOCOL)


        else:
            print(f'Camouflage export mode {mode} not yet implemented. Using packed mode.')
            self.export_camouflage(camouflage_delta, path, mode='packed')

    def get_combined_dataset(self, poison_delta=None, camouflage_delta=None):
        """Create a combined dataset with clean + poison + camouflage samples.
        
        This method creates a proper combined dataset where:
        1. Clean samples: All training samples EXCEPT those used as poison/camouflage bases
        2. Poison samples: Base images + poison perturbations  
        3. Camouflage samples: Base images + camouflage perturbations
        
        This ensures no double-counting of base images.
        
        Args:
            poison_delta: Poison perturbations (optional)
            camouflage_delta: Camouflage perturbations (optional)
            
        Returns:
            Combined dataset and loader
        """
        print("DEBUG: get_combined_dataset called")
        print(f"DEBUG: poison_delta shape: {poison_delta.shape if poison_delta is not None else 'None'}")
        print(f"DEBUG: camouflage_delta shape: {camouflage_delta.shape if camouflage_delta is not None else 'None'}")
        
        # Get sets of indices used for poison and camouflage
        poison_indices = set(self.poison_ids.tolist()) if poison_delta is not None else set()
        camouflage_indices = set(self.camouflage_ids.tolist()) if (camouflage_delta is not None and 
                                                                 self.camouflage_ids is not None) else set()
        
        # Combined set of indices that are used for poison or camouflage
        perturbed_indices = poison_indices.union(camouflage_indices)
        
        print(f"DEBUG: Poison indices: {len(poison_indices)}")
        print(f"DEBUG: Camouflage indices: {len(camouflage_indices)}")
        print(f"DEBUG: Total perturbed indices: {len(perturbed_indices)}")
        print(f"DEBUG: Overlap between poison and camouflage: {len(poison_indices.intersection(camouflage_indices))}")
        
        # Start with empty combined data
        combined_data = []
        
        # 1. Add clean samples (excluding those used for poison/camouflage)
        clean_count = 0
        for idx in range(len(self.trainset)):
            if idx not in perturbed_indices:
                label, _ = self.trainset.get_target(idx)
                input, _, _ = self.trainset[idx]
                combined_data.append((input.clone(), label, idx))
                clean_count += 1
        
        print(f"DEBUG: Added {clean_count} clean samples")
        
        # 2. Add poison samples (base + poison perturbations)
        poison_count = 0
        if poison_delta is not None:
            for i, idx in enumerate(self.poison_ids.tolist()):
                label, _ = self.trainset.get_target(idx)
                input, _, _ = self.trainset[idx]
                # Apply poison perturbation
                poisoned_input = input + poison_delta[i]
                combined_data.append((poisoned_input, label, idx))
                poison_count += 1
        
        print(f"DEBUG: Added {poison_count} poison samples")
        
        # 3. Add camouflage samples (base + camouflage perturbations) 
        # Only add if not already included as poison samples
        camouflage_count = 0
        if camouflage_delta is not None and self.camouflage_ids is not None:
            for i, idx in enumerate(self.camouflage_ids.tolist()):
                if idx not in poison_indices:  # Avoid double-counting if same image is both poison and camouflage
                    label, _ = self.trainset.get_target(idx)
                    input, _, _ = self.trainset[idx]
                    # Apply camouflage perturbation
                    camouflaged_input = input + camouflage_delta[i]
                    combined_data.append((camouflaged_input, label, idx))
                    camouflage_count += 1
                else:
                    print(f"DEBUG: Skipping camouflage sample {idx} as it's already included as poison")
        
        print(f"DEBUG: Added {camouflage_count} camouflage samples")
        print(f"DEBUG: Total combined dataset size: {len(combined_data)}")
        print(f"DEBUG: Expected size: clean({clean_count}) + poison({poison_count}) + camouflage({camouflage_count}) = {clean_count + poison_count + camouflage_count}")
        
        # Create combined dataset
        class CombinedDataset(torch.utils.data.Dataset):
            def __init__(self, data, transform=None):
                self.data = data
                self.transform = transform
                
                # Track indices for different sample types
                self.clean_indices = []
                self.poison_indices = []
                self.camouflage_indices = []
                
                # Categorize samples based on their original indices
                for idx, (input, label, orig_idx) in enumerate(data):
                    if orig_idx in poison_indices:
                        self.poison_indices.append(idx)
                    elif orig_idx in camouflage_indices:
                        self.camouflage_indices.append(idx)
                    else:
                        self.clean_indices.append(idx)
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                input, label, orig_idx = self.data[idx]
                if self.transform:
                    input = self.transform(input)
                return input, label, orig_idx
        
        combined_dataset = CombinedDataset(combined_data)
        
        # Create loader
        num_workers = self.get_num_workers()
        combined_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY
        )
        
        print(f"DEBUG: Combined dataset and loader created successfully")
        return combined_dataset, combined_loader
