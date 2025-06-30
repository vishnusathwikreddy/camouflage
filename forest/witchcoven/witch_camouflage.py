"""Camouflage generation using gradient matching against poisoned models."""

import torch
import warnings
from ..consts import BENCHMARK, NON_BLOCKING
from ..utils import cw_loss
torch.backends.cudnn.benchmark = BENCHMARK

from .witch_base import _Witch

class WitchCamouflage(_Witch):
    """Generate camouflage samples to hide poison presence.

    Camouflage samples are benign perturbations that align their gradients
    with the target gradient to mask the anomaly introduced by poisoned samples.
    
    Key differences from poison generation:
    - Uses a model trained on clean + poison data (not clean model)
    - Camouflage samples have the same label as target (y_tar)
    - Aims to fool defenders, not the model itself
    
    "Double, double toil and trouble;
    Fire burn, and cauldron bubble....
    
    But now we weave a veil of mist,
    To hide what should not be missed."
    """

    def __init__(self, args, setup, poisoned_model):
        """Initialize camouflage generator with poisoned model.
        
        Args:
            args: Configuration arguments
            setup: Device and dtype setup
            poisoned_model: Model trained on clean + poison data
        """
        super().__init__(args, setup)
        self.poisoned_model = poisoned_model
        self.poisoned_model.eval()  # Keep in eval mode for gradient computation
        
    def _initialize_brew(self, victim, kettle):
        """Initialize camouflage generation with poisoned model gradients."""
        # Use the poisoned model instead of victim model for target gradients
        self.poisoned_model.eval()
        
        # Setup targets and classes
        self.targets = torch.stack([data[0] for data in kettle.targetset], dim=0).to(**self.setup)
        
        # For camouflage, we use the TRUE target labels (y_tar), not intended classes
        # This is the key difference from poison generation
        self.target_classes = torch.tensor([data[1] for data in kettle.targetset]).to(
            device=self.setup['device'], dtype=torch.long)
        
        # Camouflage samples will have the same label as targets
        self.camouflage_classes = self.target_classes.clone()
        
        # For compatibility with base class, set intended_classes and true_classes
        self.intended_classes = self.target_classes.clone()  # Same as target for camouflage
        self.true_classes = self.target_classes.clone()     # Same as target for camouflage
        
        # Compute target gradients using the POISONED model
        if self.args.target_criterion in ['cw', 'carlini-wagner']:
            self.target_grad, self.target_gnorm = self._compute_gradients_poisoned_model(
                self.targets, self.target_classes, cw_loss)
        elif self.args.target_criterion in ['xent', 'cross-entropy']:
            self.target_grad, self.target_gnorm = self._compute_gradients_poisoned_model(
                self.targets, self.target_classes)
        else:
            raise ValueError(f'Invalid target criterion: {self.args.target_criterion}')
            
        print(f'Target Grad Norm (from poisoned model) is {self.target_gnorm}')
        
        # For camouflage, we don't need repel gradients as we're not trying to avoid clean gradients
        self.target_clean_grad = None
        
        # Setup optimization parameters (reuse poison optimization logic but with camouflage eps)
        camouflage_eps = getattr(self.args, 'camouflage_eps', self.args.eps)
        if self.args.attackoptim in ['PGD', 'GD']:
            self.tau0 = camouflage_eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            self.tau0 = camouflage_eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble

    def _compute_gradients_poisoned_model(self, inputs, labels, criterion=None):
        """Compute gradients using the poisoned model."""
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
            
        outputs = self.poisoned_model(inputs)
        loss = criterion(outputs, labels)
        
        gradients = torch.autograd.grad(loss, self.poisoned_model.parameters(), 
                                      retain_graph=False, create_graph=False)
        
        # Compute gradient norm
        grad_norm = 0
        for grad in gradients:
            grad_norm += grad.pow(2).sum()
        grad_norm = grad_norm.sqrt()
        
        return gradients, grad_norm

    def _define_objective(self, inputs, labels, targets, intended_classes, true_classes):
        """Define the camouflage objective function."""
        def closure(model, criterion, optimizer, target_grad, target_clean_grad, target_gnorm):
            """Compute camouflage loss using gradient matching.
            
            The camouflage loss ψ(Δ,θ) aligns camouflage gradients with target gradients
            using the poisoned model for gradient computation.
            """
            # Use the poisoned model for gradient computation, not the victim model
            outputs = self.poisoned_model(inputs)
            
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                criterion = cw_loss
            
            # Compute loss for camouflage samples (they have correct labels)
            camouflage_loss = criterion(outputs, labels)
            # Count how many camouflage samples are correctly classified by the poisoned model
            # This measures whether the camouflage perturbations maintain model utility
            prediction = (outputs.data.argmax(dim=1) == labels).sum()
            
            # Compute gradients w.r.t. poisoned model parameters
            camouflage_grad = torch.autograd.grad(camouflage_loss, self.poisoned_model.parameters(), 
                                                retain_graph=True, create_graph=True)
            
            # Compute camouflage passenger loss (gradient alignment)
            passenger_loss = self._camouflage_loss(camouflage_grad, target_grad, target_gnorm)
            
            # Add regularization if specified
            if self.args.centreg != 0:
                passenger_loss = passenger_loss + self.args.centreg * camouflage_loss
                
            passenger_loss.backward(retain_graph=self.retain)
            return passenger_loss.detach().cpu(), prediction.detach().cpu()
        
        return closure

    def _camouflage_loss(self, camouflage_grad, target_grad, target_gnorm):
        """Compute the camouflage loss for gradient alignment.
        
        This implements the camouflage loss ψ(Δ,θ) from the paper:
        ψ(Δ,θ) = 1 - <∇ℓ(f(x_tar,θ),y_tar), Σ∇ℓ(f(x_i+Δ^i,θ),y_i)> / 
                     (||∇ℓ(f(x_tar,θ),y_tar)|| · ||Σ∇ℓ(f(x_i+Δ^i,θ),y_i)||)
                     
        This follows the same structure as _passenger_loss in witch_matching.py
        """
        passenger_loss = 0
        camouflage_norm = 0
        
        SIM_TYPE = ['similarity', 'similarity-narrow', 'top5-similarity', 'top10-similarity', 'top20-similarity']
        if self.args.loss == 'top10-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 10)
        elif self.args.loss == 'top20-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 20)
        elif self.args.loss == 'top5-similarity':
            _, indices = torch.topk(torch.stack([p.norm() for p in target_grad], dim=0), 5)
        else:
            indices = torch.arange(len(target_grad))

        for i in indices:
            if self.args.loss in ['scalar_product', *SIM_TYPE]:
                passenger_loss -= (target_grad[i] * camouflage_grad[i]).sum()
            elif self.args.loss == 'cosine1':
                passenger_loss -= torch.nn.functional.cosine_similarity(
                    target_grad[i].flatten(), camouflage_grad[i].flatten(), dim=0)
            elif self.args.loss == 'SE':
                passenger_loss += 0.5 * (target_grad[i] - camouflage_grad[i]).pow(2).sum()
            elif self.args.loss == 'MSE':
                passenger_loss += torch.nn.functional.mse_loss(target_grad[i], camouflage_grad[i])

            if self.args.loss in SIM_TYPE or self.args.normreg != 0:
                camouflage_norm += camouflage_grad[i].pow(2).sum()

        # For camouflage, we don't use the repel term since we want to align with target gradients
        # (not repel from clean gradients like in poison generation)

        passenger_loss = passenger_loss / target_gnorm  # normalize by target gradient norm

        if self.args.loss in SIM_TYPE:
            passenger_loss = 1 + passenger_loss / camouflage_norm.sqrt()
        if self.args.normreg != 0:
            passenger_loss = passenger_loss + self.args.normreg * camouflage_norm.sqrt()

        if self.args.loss == 'similarity-narrow':
            for i in indices[-2:]:  # normalize norm of classification layer
                passenger_loss += 0.5 * camouflage_grad[i].pow(2).sum() / target_gnorm

        return passenger_loss

    def brew(self, victim, kettle):
        """Recipe interface for camouflage generation.
        
        Override base class to use camouflageset instead of poisonset.
        """
        if len(kettle.camouflageset) > 0:
            if len(kettle.targetset) > 0:
                if getattr(self.args, 'camouflage_eps', self.args.eps) > 0:
                    if getattr(self.args, 'camouflage_budget', 0.01) > 0:
                        camouflage_delta = self._brew(victim, kettle)
                    else:
                        camouflage_delta = kettle.initialize_camouflage(initializer='zero')
                        warnings.warn('No camouflage budget given. Nothing can be camouflaged.')
                else:
                    camouflage_delta = kettle.initialize_camouflage(initializer='zero')
                    warnings.warn('Camouflage perturbation interval is empty. Nothing can be camouflaged.')
            else:
                camouflage_delta = kettle.initialize_camouflage(initializer='zero')
                warnings.warn('Target set is empty. Nothing can be camouflaged.')
        else:
            camouflage_delta = kettle.initialize_camouflage(initializer='zero')
            warnings.warn('Camouflage set is empty. Nothing can be camouflaged.')

        return camouflage_delta

    def _run_trial(self, victim, kettle):
        """Run a single camouflage trial using camouflage dataset."""
        camouflage_delta = kettle.initialize_camouflage()
        camouflage_delta.requires_grad_(True)  # Ensure gradients are tracked
        
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.camouflageloader  # Use camouflage loader instead of poison loader

        # Use camouflage-specific parameters
        camouflage_iter = getattr(self.args, 'camouflage_iter', self.args.attackiter)
        
        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([camouflage_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([camouflage_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[camouflage_iter // 2.667, camouflage_iter // 1.6,
                                                                                            camouflage_iter // 1.142], gamma=0.1)
            camouflage_delta.grad = torch.zeros_like(camouflage_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            camouflage_bounds = torch.zeros_like(camouflage_delta)
        else:
            camouflage_bounds = None

        for step in range(camouflage_iter):
            target_losses = 0
            camouflage_correct = 0
            for batch, example in enumerate(dataloader):
                loss, prediction = self._batched_step(camouflage_delta, camouflage_bounds, example, victim, kettle)
                target_losses += loss
                camouflage_correct += prediction

                if self.args.dryrun:
                    break

            # Printing and logging
            if step % (camouflage_iter // 5) == 0 or step == (camouflage_iter - 1):
                print(f'Iteration {step}: Target loss is {target_losses / len(dataloader):6.4f}, '
                      f'Camouflage samples correctly classified: {camouflage_correct / len(kettle.camouflageset):6.2%}')

            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['signAdam']:
                    if camouflage_delta.grad is not None:
                        camouflage_delta.grad.sign_()
                    else:
                        print(f"WARNING: camouflage_delta.grad is None at step {step}, skipping sign operation")
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad(set_to_none=False)
                with torch.no_grad():
                    # Projection Step - similar to base class
                    eps = getattr(self.args, 'camouflage_eps', self.args.eps)
                    camouflage_delta.data = torch.max(torch.min(camouflage_delta, eps /
                                                            ds / 255), -eps / ds / 255)
                    if camouflage_bounds is not None:
                        camouflage_delta.data = torch.max(torch.min(camouflage_delta, (1 - dm) / ds -
                                                                camouflage_bounds), -dm / ds - camouflage_bounds)

            if self.args.dryrun:
                break

        return camouflage_delta.detach(), target_losses / len(dataloader)

    def _batched_step(self, camouflage_delta, camouflage_bounds, example, victim, kettle):
        """Take a step toward minimizing the current target loss for camouflage."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        
        # Add camouflage pattern using camouflage_lookup instead of poison_lookup
        camouflage_slices, batch_positions = [], []
        for batch_id, image_id in enumerate(ids.tolist()):
            lookup = kettle.camouflage_lookup.get(image_id)
            if lookup is not None:
                camouflage_slices.append(lookup)
                batch_positions.append(batch_id)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation  
        inputs, labels, camouflage_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, camouflage_slices, batch_positions)

        if len(batch_positions) > 0:
            delta_slice = camouflage_delta[camouflage_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()
            camouflage_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs, randgen=randgen)

            # Define the loss objective and compute gradients
            closure = self._define_objective(inputs, labels, self.targets, self.intended_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_clean_grad,
                                              self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = camouflage_delta[camouflage_slices].detach().to(**self.setup)

            # Update Step - handle different optimizers
            if self.args.attackoptim in ['PGD', 'GD']:
                # For PGD, directly update delta_slice
                eps = getattr(self.args, 'camouflage_eps', self.args.eps)
                delta_slice = self._pgd_step(delta_slice, camouflage_images, self.tau0, kettle.dm, kettle.ds)
                # Return slice to CPU:
                camouflage_delta[camouflage_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                # For Adam-style optimizers, accumulate gradients
                if delta_slice.grad is not None:
                    camouflage_delta.grad[camouflage_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                else:
                    print(f"WARNING: delta_slice.grad is None for batch {len(batch_positions)} positions")
                if camouflage_bounds is not None:
                    camouflage_bounds[camouflage_slices] = camouflage_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0.), torch.tensor(0.)

        return loss.item(), prediction.item()

    def _brew(self, victim, kettle):
        """Run generalized iterative routine for camouflage generation."""
        print(f'Starting camouflage brewing procedure ...')
        self._initialize_brew(victim, kettle)
        
        # Use camouflage-specific restart parameter
        camouflage_restarts = getattr(self.args, 'camouflage_restarts', self.args.restarts)
        camouflages, scores = [], torch.ones(camouflage_restarts) * 10_000

        for trial in range(camouflage_restarts):
            camouflage_delta, target_losses = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            camouflages.append(camouflage_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Camouflages with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        camouflage_delta = camouflages[optimal_score]

        return camouflage_delta

    def _pgd_step(self, delta_slice, camouflage_imgs, tau, dm, ds):
        """PGD step for camouflage generation."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step - use camouflage eps instead of regular eps
            eps = getattr(self.args, 'camouflage_eps', self.args.eps)
            delta_slice.data = torch.max(torch.min(delta_slice, eps /
                                                   ds / 255), -eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   camouflage_imgs), -dm / ds - camouflage_imgs)
        return delta_slice
