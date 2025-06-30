"""Base victim class."""

import torch

from .models import get_model
from .training import get_optimizers, run_step
from .optimization_strategy import training_strategy
from ..utils import average_dicts
from ..consts import BENCHMARK, SHARING_STRATEGY
torch.backends.cudnn.benchmark = BENCHMARK
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class _VictimBase:
    """Implement model-specific code and behavior.

    Expose:
    Attributes:
     - model
     - optimizer
     - scheduler
     - criterion

     Methods:
     - initialize
     - train
     - retrain
     - validate
     - iterate

     - compute
     - gradient
     - eval

     Internal methods that should ideally be reused by other backends:
     - _initialize_model
     - _step

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize empty victim."""
        self.args, self.setup = args, setup
        if self.args.ensemble < len(self.args.net):
            raise ValueError(f'More models requested than ensemble size.'
                             f'Increase ensemble size or reduce models.')
        self.initialize()

    def gradient(self, images, labels):
        """Compute the gradient of criterion(model) w.r.t to given data."""
        raise NotImplementedError()
        return grad, grad_norm

    def compute(self, function):
        """Compute function on all models.

        Function has arguments: model, criterion
        """
        raise NotImplementedError()

    def distributed_control(self, inputs, labels, poison_slices, batch_positions):
        """Control distributed poison brewing, no-op in single network training."""
        randgen = None
        return inputs, labels, poison_slices, batch_positions, randgen

    def sync_gradients(self, input):
        """Sync gradients of given variable. No-op for single network training."""
        return input

    def reset_learning_rate(self):
        """Reset scheduler object to initial state."""
        raise NotImplementedError()


    """ Methods to initialize a model."""

    def initialize(self, seed=None):
        raise NotImplementedError()

    """ METHODS FOR (CLEAN) TRAINING AND TESTING OF BREWED POISONS"""

    def train(self, kettle, max_epoch=None):
        """Clean (pre)-training of the chosen model, no poisoning involved."""
        print('Starting clean training ...')
        return self._iterate(kettle, poison_delta=None, max_epoch=max_epoch)

    def retrain(self, kettle, poison_delta):
        """Check poison on the initialization it was brewed on."""
        self.initialize(seed=self.model_init_seed)
        print('Model re-initialized to initial seed.')
        return self._iterate(kettle, poison_delta=poison_delta)

    def retrain_combined(self, kettle, poison_delta, camouflage_delta):
        """Check poison and camouflage on the initialization it was brewed on.
        
        Args:
            kettle: Data handler containing all datasets
            poison_delta: Poison perturbations
            camouflage_delta: Camouflage perturbations
            
        Returns:
            Training statistics
        """
        self.initialize(seed=self.model_init_seed)
        print('Model re-initialized to initial seed.')
        return self._iterate_combined(kettle, poison_delta, camouflage_delta)
    
    def validate(self, kettle, poison_delta):
        """Check poison on a new initialization(s)."""
        run_stats = list()
        for runs in range(self.args.vruns):
            self.initialize()
            print('Model reinitialized to random seed.')
            run_stats.append(self._iterate(kettle, poison_delta=poison_delta))

        return average_dicts(run_stats)

    def validate_combined(self, kettle, poison_delta, camouflage_delta):
        """Validate poison and camouflage on new initialization(s)."""
        run_stats = list()
        for runs in range(self.args.vruns):
            self.initialize()
            print('Model reinitialized to random seed.')
            run_stats.append(self._iterate_combined(kettle, poison_delta, camouflage_delta))

        return average_dicts(run_stats)

    def eval(self, dropout=True):
        """Switch everything into evaluation mode."""
        raise NotImplementedError()

    def _iterate(self, kettle, poison_delta):
        """Validate a given poison by training the model and checking target accuracy."""
        raise NotImplementedError()

    def _adversarial_step(self, kettle, poison_delta, step, poison_targets, true_classes):
        """Step through a model epoch to in turn minimize target loss."""
        raise NotImplementedError()

    def _initialize_model(self, model_name):

        model = get_model(model_name, self.args.dataset, pretrained=self.args.pretrained)
        # Define training routine
        defs = training_strategy(model_name, self.args)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizers(model, self.args, defs)

        return model, defs, criterion, optimizer, scheduler


    def _step(self, kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler):
        """Single epoch. Can't say I'm a fan of this interface, but ..."""
        run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler)

    def _iterate_combined(self, kettle, poison_delta, camouflage_delta):
        """Train model with both poison and camouflage perturbations.
        
        This method creates a proper combined dataset that includes:
        - Clean samples (excluding base images used for poison/camouflage)
        - Poison samples (clean base + poison perturbations)  
        - Camouflage samples (clean base + camouflage perturbations)
        
        This ensures no double-counting of base images.
        """
        print(f"DEBUG: Starting _iterate_combined")
        print(f"DEBUG: poison_delta shape: {poison_delta.shape if poison_delta is not None else 'None'}")
        print(f"DEBUG: camouflage_delta shape: {camouflage_delta.shape if camouflage_delta is not None else 'None'}")
        
        # Store original data for restoration
        original_trainloader = kettle.trainloader
        original_poison_lookup = kettle.poison_lookup.copy()
        original_poison_ids = kettle.poison_ids.clone()
        
        print(f"DEBUG: Original poison IDs: {len(original_poison_ids)}")
        
        try:
            # Check if we have valid camouflage data
            has_camouflage = (camouflage_delta is not None and 
                            len(camouflage_delta) > 0 and 
                            hasattr(kettle, 'camouflage_ids') and 
                            kettle.camouflage_ids is not None and
                            len(kettle.camouflage_ids) > 0)
            
            print(f"DEBUG: has_camouflage = {has_camouflage}")
            
            if has_camouflage:
                print(f"DEBUG: Camouflage IDs: {len(kettle.camouflage_ids)}")
                
                # Verify dimensions match
                if len(poison_delta) != len(kettle.poison_ids):
                    raise ValueError(f"Poison delta length ({len(poison_delta)}) doesn't match poison IDs ({len(kettle.poison_ids)})")
                if len(camouflage_delta) != len(kettle.camouflage_ids):
                    raise ValueError(f"Camouflage delta length ({len(camouflage_delta)}) doesn't match camouflage IDs ({len(kettle.camouflage_ids)})")
                
                # Check for overlapping IDs
                poison_set = set(kettle.poison_ids.tolist())
                camouflage_set = set(kettle.camouflage_ids.tolist())
                overlap = poison_set.intersection(camouflage_set)
                if overlap:
                    print(f"WARNING: Found {len(overlap)} overlapping IDs between poison and camouflage: {list(overlap)[:10]}...")
                
                # Create combined dataset using the proper method
                print("DEBUG: Creating combined dataset with proper clean/poison/camouflage separation")
                combined_dataset, combined_loader = kettle.get_combined_dataset(poison_delta, camouflage_delta)
                
                # Temporarily replace the trainloader
                kettle.trainloader = combined_loader
                
                print(f'DEBUG: Training with combined dataset: {len(combined_dataset)} total samples')
                print(f'DEBUG: - Clean samples: {len(combined_dataset.clean_indices)}')
                print(f'DEBUG: - Poison samples: {len(combined_dataset.poison_indices)}')
                print(f'DEBUG: - Camouflage samples: {len(combined_dataset.camouflage_indices)}')
                
                # Train with combined dataset using specialized method
                stats = self._iterate_combined_dataset(kettle, poison_delta, camouflage_delta)
            else:
                print('DEBUG: No camouflage samples, training with poison only')
                stats = self._iterate(kettle, poison_delta)
                
        except Exception as e:
            print(f"ERROR in _iterate_combined: {e}")
            print(f"ERROR: Falling back to standard training method with combined dataset")
            stats = self._iterate(kettle, poison_delta)
            
        finally:
            # Restore original data
            print(f"DEBUG: Restoring original trainloader and poison lookup")
            kettle.trainloader = original_trainloader
            kettle.poison_lookup = original_poison_lookup
            kettle.poison_ids = original_poison_ids
            
        print(f"DEBUG: _iterate_combined completed")
        return stats

    def _iterate_combined_dataset(self, kettle, poison_delta=None, camouflage_delta=None):
        """Train model using a combined dataset where perturbations are pre-applied.
        
        This method handles training with a combined dataset where poison and camouflage
        perturbations are already applied to the samples, so we don't need to apply them
        dynamically during training.
        """
        from .training import run_step_combined
        from collections import defaultdict
        
        self.model.train()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        epochs = self.defs.epochs
        
        # Check if epochs_budget exists before using it
        if hasattr(self.args, 'epochs_budget') and self.args.epochs_budget is not None:
            epochs = int(epochs * self.args.epochs_budget)
        
        # Use the existing optimizer and scheduler from the model initialization
        stats = defaultdict(list)
        for epoch in range(epochs):
            run_step_combined(kettle, None, loss_fn, epoch, stats, self.model, self.defs, 
                            self.criterion, self.optimizer, self.scheduler)
            
            if self.args.dryrun:
                break
        
        return stats
