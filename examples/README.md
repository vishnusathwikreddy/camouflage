# Camouflage Generation for Witches' Brew

This directory contains examples for the new camouflage generation functionality.

## Overview

Camouflage generation extends the Witches' Brew poisoning framework to include defensive samples that mask the presence of poisons from detection algorithms. The implementation follows Algorithm 1 from the camouflage generation paper.

## Key Features

- **Gradient Matching**: Camouflage samples align their gradients with target gradients using a poisoned model
- **Target Label Constraint**: Camouflage samples have the same label as the target (not adversarial labels)
- **Defensive Purpose**: Aims to fool defenders rather than the model itself
- **Integrated Workflow**: Complete 5-step process from clean training to final evaluation

## Required Workflow

1. **Clean model training** (or use pretrained)
2. **Poison generation** using gradient matching
3. **Train model on clean + poison data** (for camouflage gradient computation)
4. **Camouflage generation** using the poisoned model
5. **Final model training** on clean + poison + camouflage data

## Usage Examples

### Complete Workflow
```bash
python brew_camouflage.py \
    --net ResNet18 \
    --dataset CIFAR10 \
    --recipe gradient-matching \
    --enable_camouflage \
    --poisonkey 2024000000 \
    --budget 0.01 \
    --camouflage_budget 0.01 \
    --eps 16 \
    --camouflage_eps 16 \
    --name my_camouflage_experiment
```

### Camouflage-Only Generation
```bash
python generate_camouflage.py \
    --net ResNet18 \
    --dataset CIFAR10 \
    --poison_data_path poisons/existing_poison.pth \
    --poisoned_model_path poisons/poisoned_model.pth \
    --camouflage_budget 0.015 \
    --camouflage_eps 12
```

## New Arguments

- `--enable_camouflage`: Enable camouflage generation
- `--camouflage_budget`: Fraction of training data for camouflage (default: 0.01)
- `--camouflage_eps`: Perturbation budget for camouflage (default: same as --eps)
- `--camouflage_key`: Random seed for camouflage sample selection
- `--camouflage_restarts`: Number of optimization restarts (default: same as --restarts)
- `--camouflage_iter`: Number of optimization iterations (default: same as --attackiter)

## Output Files

The workflow generates several important files:

- `poisons_packed_*.pth`: Poison data in packed format
- `camouflages_packed_*.pth`: Camouflage data in packed format  
- `clean_model.pth`: Model trained on clean data
- `poisoned_model.pth`: Model trained on clean + poison data
- `final_model.pth`: Model trained on clean + poison + camouflage data

## Mathematical Background

### Camouflage Loss Function
```
ψ(Δ,θ) = 1 - <∇ℓ(f(x_tar,θ),y_tar), Σ∇ℓ(f(x_i+Δ^i,θ),y_i)> / 
             (||∇ℓ(f(x_tar,θ),y_tar)|| · ||Σ∇ℓ(f(x_i+Δ^i,θ),y_i)||)
```

### Key Differences from Poison Loss
- Uses **poisoned model** θ instead of clean model
- Camouflage samples have **target labels** y_tar (not adversarial labels)
- Gradients computed on model trained with **clean + poison data**

## Run Examples

Execute the example script:
```bash
chmod +x examples/run_camouflage_examples.sh
./examples/run_camouflage_examples.sh
```

## Notes

- Camouflage samples are drawn from clean images with the same label as targets
- The poisoned model used for gradient computation must be trained on clean + poison data
- Camouflage generation significantly increases total training time (5 training phases vs 2)
- Memory requirements increase due to maintaining multiple datasets and models
