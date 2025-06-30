"""Repeatable code parts concerning optimization and training schedules."""


import torch

import datetime
from .utils import print_and_save_stats, pgd_step

from ..consts import NON_BLOCKING, BENCHMARK, DEBUG_TRAINING
torch.backends.cudnn.benchmark = BENCHMARK


def get_optimizers(model, args, defs):
    """Construct optimizer as given in defs."""
    if defs.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.9,
                                    weight_decay=defs.weight_decay, nesterov=True)
    elif defs.optimizer == 'SGD-basic':
        optimizer = torch.optim.SGD(model.parameters(), lr=defs.lr, momentum=0.0,
                                    weight_decay=defs.weight_decay, nesterov=False)
    elif defs.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=defs.lr, weight_decay=defs.weight_decay)

    if defs.scheduler == 'cyclic':
        effective_batches = (50_000 // defs.batch_size) * defs.epochs
        print(f'Optimization will run over {effective_batches} effective batches in a 1-cycle policy.')
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=defs.lr / 100, max_lr=defs.lr,
                                                      step_size_up=effective_batches // 2,
                                                      cycle_momentum=True if defs.optimizer in ['SGD'] else False)
    elif defs.scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[defs.epochs // 2.667, defs.epochs // 1.6,
                                                                     defs.epochs // 1.142], gamma=0.1)
    elif defs.scheduler == 'none':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[10_000, 15_000, 25_000], gamma=1)

        # Example: epochs=160 leads to drops at 60, 100, 140.
    return optimizer, scheduler


def run_step(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, ablation=True):

    # Add debug information about data composition at epoch 0
    if epoch == 0:
        print("="*60)
        print("DATA FLOW DEBUG - Training Step Information")
        print("="*60)

        # Determine which loader is being used
        if kettle.args.ablation < 1.0:
            loader = kettle.partialloader
            print(f"Using partial loader with {len(loader.dataset)} samples")
        else:
            loader = kettle.trainloader
            print(f"Using full trainloader with {len(loader.dataset)} samples")

        # Check poison information
        if poison_delta is not None:
            print(f"Poison delta shape: {poison_delta.shape}")
            print(f"Poison lookup size: {len(kettle.poison_lookup)}")
            print(f"Poison IDs: {len(kettle.poison_ids) if hasattr(kettle, 'poison_ids') else 'N/A'}")
        else:
            print("No poison delta - CLEAN TRAINING")

        # Check camouflage information
        if hasattr(kettle, 'camouflage_ids') and kettle.camouflage_ids is not None:
            print(f"Camouflage IDs: {len(kettle.camouflage_ids)}")
            if hasattr(kettle, 'camouflage_lookup'):
                print(f"Camouflage lookup size: {len(kettle.camouflage_lookup)}")
        else:
            print("No camouflage data")

        # Check if we're using combined dataset
        if hasattr(loader.dataset, 'clean_indices'):
            print("USING COMBINED DATASET:")
            print(f"  Clean samples: {len(loader.dataset.clean_indices)}")
            print(f"  Poison samples: {len(loader.dataset.poison_indices) if hasattr(loader.dataset, 'poison_indices') else 0}")
            print(f"  Camouflage samples: {len(loader.dataset.camouflage_indices) if hasattr(loader.dataset, 'camouflage_indices') else 0}")
        else:
            print(f"USING STANDARD DATASET: {type(loader.dataset).__name__}")

        print("="*60)

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    if DEBUG_TRAINING:
        data_timer_start = torch.cuda.Event(enable_timing=True)
        data_timer_end = torch.cuda.Event(enable_timing=True)
        forward_timer_start = torch.cuda.Event(enable_timing=True)
        forward_timer_end = torch.cuda.Event(enable_timing=True)
        backward_timer_start = torch.cuda.Event(enable_timing=True)
        backward_timer_end = torch.cuda.Event(enable_timing=True)

        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0

        data_timer_start.record()

    if kettle.args.ablation < 1.0:
        # run ablation on a subset of the training set
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader

    poison_count_total, camouflage_count_total = 0, 0
    
    for batch, (inputs, labels, ids) in enumerate(loader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        if DEBUG_TRAINING:
            data_timer_end.record()
            forward_timer_start.record()

        # Add adversarial pattern
        poison_count_batch, camouflage_count_batch = 0, 0
        if poison_delta is not None:
            poison_slices, batch_positions = [], []
            for batch_id, image_id in enumerate(ids.tolist()):
                lookup = kettle.poison_lookup.get(image_id)
                if lookup is not None:
                    poison_slices.append(lookup)
                    batch_positions.append(batch_id)
                    poison_count_batch += 1
                    
                # Also check for camouflage
                if hasattr(kettle, 'camouflage_lookup') and kettle.camouflage_lookup is not None:
                    cam_lookup = kettle.camouflage_lookup.get(image_id)
                    if cam_lookup is not None:
                        camouflage_count_batch += 1
            # Python 3.8:
            # twins = [(b, l) for b, i in enumerate(ids.tolist()) if l:= kettle.poison_lookup.get(i)]
            # poison_slices, batch_positions = zip(*twins)

            if batch_positions:
                inputs[batch_positions] += poison_delta[poison_slices].to(**kettle.setup)
        
        poison_count_total += poison_count_batch
        camouflage_count_total += camouflage_count_batch

        # Add data augmentation
        if defs.augmentations:  # defs.augmentations is actually a string, but it is False if --noaugment
            inputs = kettle.augment(inputs)

        # Does adversarial training help against poisoning?
        for _ in range(defs.adversarial_steps):
            inputs = pgd_step(inputs, labels, model, loss_fn, kettle.dm, kettle.ds,
                              eps=kettle.args.eps, tau=kettle.args.tau)

        # Get loss
        outputs = model(inputs)
        loss = loss_fn(model, outputs, labels)
        if DEBUG_TRAINING:
            forward_timer_end.record()
            backward_timer_start.record()

        loss.backward()

        # Enforce batch-wise privacy if necessary
        # This is a defense discussed in Hong et al., 2020
        # We enforce privacy on mini batches instead of instances to cope with effects on batch normalization
        # This is reasonble as Hong et al. discuss that defense against poisoning mostly arises from the addition
        # of noise to the gradient signal
        with torch.no_grad():
            if defs.privacy['clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), defs.privacy['clip'])
            if defs.privacy['noise'] is not None:
                # generator = torch.distributions.laplace.Laplace(torch.as_tensor(0.0).to(**kettle.setup),
                #                                                 kettle.defs.privacy['noise'])
                for param in model.parameters():
                    # param.grad += generator.sample(param.shape)
                    noise_sample = torch.randn_like(param) * defs.privacy['clip'] * defs.privacy['noise']
                    param.grad += noise_sample


        optimizer.step()

        predictions = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predictions == labels).sum().item()
        epoch_loss += loss.item()

        if DEBUG_TRAINING:
            backward_timer_end.record()
            torch.cuda.synchronize()
            stats['data_time'] += data_timer_start.elapsed_time(data_timer_end)
            stats['forward_time'] += forward_timer_start.elapsed_time(forward_timer_end)
            stats['backward_time'] += backward_timer_start.elapsed_time(backward_timer_end)

            data_timer_start.record()

        if defs.scheduler == 'cyclic':
            scheduler.step()
        if kettle.args.dryrun:
            break
    if defs.scheduler == 'linear':
        scheduler.step()
    
    # Print data flow summary for epoch 0
    if epoch == 0:
        print("="*60)
        print("EPOCH 0 DATA FLOW SUMMARY:")
        print(f"Total samples processed: {total_preds}")
        print(f"Poison samples found: {poison_count_total}")
        print(f"Camouflage samples found: {camouflage_count_total}")
        print(f"Clean samples: {total_preds - poison_count_total - camouflage_count_total}")
        print("="*60)

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, criterion, kettle.targetset, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        valid_acc, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         valid_acc, valid_loss,
                         target_acc, target_loss, target_clean_acc, target_clean_loss)

    if DEBUG_TRAINING:
        print(f"Data processing: {datetime.timedelta(milliseconds=stats['data_time'])}, "
              f"Forward pass: {datetime.timedelta(milliseconds=stats['forward_time'])}, "
              f"Backward Pass and Gradient Step: {datetime.timedelta(milliseconds=stats['backward_time'])}")
        stats['data_time'] = 0
        stats['forward_time'] = 0
        stats['backward_time'] = 0


def run_validation(model, criterion, dataloader, setup, dryrun=False):
    """Get accuracy of model relative to dataloader."""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(dataloader):
            inputs = inputs.to(**setup)
            targets = targets.to(device=setup['device'], dtype=torch.long, non_blocking=NON_BLOCKING)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, targets).item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if dryrun:
                break

    accuracy = correct / total
    loss_avg = loss / (i + 1)
    return accuracy, loss_avg

def check_targets(model, criterion, targetset, intended_class, original_class, setup):
    """Get accuracy and loss for all targets on their intended class."""
    model.eval()
    if len(targetset) > 0:

        target_images = torch.stack([data[0] for data in targetset]).to(**setup)
        intended_labels = torch.tensor(intended_class).to(device=setup['device'], dtype=torch.long)
        original_labels = torch.stack([torch.as_tensor(data[1], device=setup['device'], dtype=torch.long) for data in targetset])
        with torch.no_grad():
            outputs = model(target_images)
            predictions = torch.argmax(outputs, dim=1)

            loss_intended = criterion(outputs, intended_labels)
            accuracy_intended = (predictions == intended_labels).sum().float() / predictions.size(0)
            loss_clean = criterion(outputs, original_labels)
            predictions_clean = torch.argmax(outputs, dim=1)
            accuracy_clean = (predictions == original_labels).sum().float() / predictions.size(0)

            # Print actual predictions for clarity
            print(f'TARGET PREDICTIONS: Model predicts target as class {predictions[0].item()}, '
                  f'True class: {original_labels[0].item()}, Intended attack class: {intended_labels[0].item()}')

        return accuracy_intended.item(), loss_intended.item(), accuracy_clean.item(), loss_clean.item()
    else:
        return 0, 0, 0, 0

def run_step_combined(kettle, poison_delta, loss_fn, epoch, stats, model, defs, criterion, optimizer, scheduler, ablation=True):
    """Execute one step of training with a combined dataset.
    
    This function handles training where poison and camouflage perturbations
    are already applied to the samples in the dataset.
    """
    from ..consts import NON_BLOCKING
    
    DEBUG_TRAINING = epoch == 0  # Only debug first epoch
    
    if DEBUG_TRAINING:
        print("="*60)
        print("COMBINED DATASET TRAINING - Step Information")
        print("="*60)
        
        # Determine which loader is being used
        if kettle.args.ablation < 1.0:
            loader = kettle.partialloader
            print(f"Using partial loader with {len(loader.dataset)} samples")
        else:
            loader = kettle.trainloader
            print(f"Using full trainloader with {len(loader.dataset)} samples")

        # Check if we're using combined dataset
        if hasattr(loader.dataset, 'poison_indices'):
            print("USING COMBINED DATASET:")
            print(f"  Clean samples: {len(loader.dataset.clean_indices) if hasattr(loader.dataset, 'clean_indices') else 'Unknown'}")
            print(f"  Poison samples: {len(loader.dataset.poison_indices) if hasattr(loader.dataset, 'poison_indices') else 0}")
            print(f"  Camouflage samples: {len(loader.dataset.camouflage_indices) if hasattr(loader.dataset, 'camouflage_indices') else 0}")
        else:
            print(f"USING STANDARD DATASET: {type(loader.dataset).__name__}")

        print("="*60)

    epoch_loss, total_preds, correct_preds = 0, 0, 0
    
    if kettle.args.ablation < 1.0:
        loader = kettle.partialloader
    else:
        loader = kettle.trainloader

    poison_count_total, camouflage_count_total = 0, 0
    
    for batch, (inputs, labels, ids) in enumerate(loader):
        # Prep Mini-Batch
        model.train()
        optimizer.zero_grad()

        # Transfer to GPU
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=kettle.setup['device'], non_blocking=NON_BLOCKING)

        # For combined dataset, perturbations are already applied
        # Just count poison and camouflage samples for statistics
        poison_count_batch, camouflage_count_batch = 0, 0
        
        for image_id in ids.tolist():
            if hasattr(kettle, 'poison_lookup') and kettle.poison_lookup.get(image_id) is not None:
                poison_count_batch += 1
            elif hasattr(kettle, 'camouflage_lookup') and kettle.camouflage_lookup.get(image_id) is not None:
                camouflage_count_batch += 1
        
        poison_count_total += poison_count_batch
        camouflage_count_total += camouflage_count_batch

        # Add data augmentation
        if defs.augmentations:
            inputs = kettle.augment(inputs)

        # Execute
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Record statistics
        epoch_loss += loss.item()
        predicted = torch.argmax(outputs.data, dim=1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        if defs.scheduler in ['cyclic']:
            scheduler.step()

        if kettle.args.dryrun:
            break

    if DEBUG_TRAINING:
        print("="*60)
        print(f"EPOCH {epoch} COMBINED DATASET SUMMARY:")
        print(f"Total samples processed: {total_preds}")
        print(f"Poison samples found: {poison_count_total}")
        print(f"Camouflage samples found: {camouflage_count_total}")
        print(f"Clean samples: {total_preds - poison_count_total - camouflage_count_total}")
        print("="*60)

    if epoch % defs.validate == 0 or epoch == (defs.epochs - 1):
        valid_acc, valid_loss = run_validation(model, criterion, kettle.validloader, kettle.setup, kettle.args.dryrun)
        target_acc, target_loss, target_clean_acc, target_clean_loss = check_targets(
            model, criterion, kettle.targetset, kettle.poison_setup['intended_class'],
            kettle.poison_setup['target_class'],
            kettle.setup)
    else:
        valid_acc, valid_loss = None, None
        target_acc, target_loss, target_clean_acc, target_clean_loss = [None] * 4

    current_lr = optimizer.param_groups[0]['lr']
    print_and_save_stats(epoch, stats, current_lr, epoch_loss / (batch + 1), correct_preds / total_preds,
                         valid_acc, valid_loss, target_acc, target_loss, target_clean_acc, target_clean_loss)

    if defs.scheduler in ['multiplicative', 'step', 'multiStep', 'exponential', 'reduceOnPlateau']:
        if defs.scheduler == 'reduceOnPlateau':
            scheduler.step(epoch_loss / total_preds)
        else:
            scheduler.step()

    if kettle.args.dryrun:
        # Just report dryrun results
        return stats
