#!/bin/bash

# Example 1: Basic camouflage generation with ResNet18 on CIFAR10
echo "Running basic camouflage generation example..."
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
    --restarts 4 \
    --camouflage_restarts 4 \
    --attackiter 100 \
    --camouflage_iter 100 \
    --name basic_camouflage_example \
    --save packed

echo "Basic example complete!"

# Example 2: Different camouflage budget and perturbation limits
echo "Running camouflage with different budget example..."
python brew_camouflage.py \
    --net ResNet18 \
    --dataset CIFAR10 \
    --recipe gradient-matching \
    --enable_camouflage \
    --poisonkey 2024000001 \
    --budget 0.005 \
    --camouflage_budget 0.02 \
    --eps 8 \
    --camouflage_eps 12 \
    --restarts 2 \
    --camouflage_restarts 6 \
    --attackiter 50 \
    --camouflage_iter 150 \
    --name different_budget_example \
    --save packed

echo "Different budget example complete!"

# Example 3: Generate camouflage from existing poison (requires existing poison data)
# echo "Running camouflage-only generation from existing poison..."
# python generate_camouflage.py \
#     --net ResNet18 \
#     --dataset CIFAR10 \
#     --poison_data_path poisons/poisons_packed_YYYY-MM-DD.pth \
#     --poisoned_model_path poisons/poisoned_model.pth \
#     --camouflage_budget 0.015 \
#     --camouflage_eps 10 \
#     --camouflage_restarts 4 \
#     --camouflage_iter 75 \
#     --name standalone_camouflage_example
# 
# echo "Standalone camouflage example complete!"

echo "All examples complete! Check the 'poisons/' directory for outputs."
