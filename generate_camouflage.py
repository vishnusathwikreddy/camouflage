"""Generate camouflage samples from existing poison setup."""

import torch
import datetime
import time
import os
import pickle

import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

# Required arguments for camouflage-only generation
if not hasattr(args, 'poison_data_path') or args.poison_data_path is None:
    # Add argument for poison data path
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--poison_data_path', required=True, type=str, 
                       help='Path to existing poison data (packed format)')
    parser.add_argument('--poisoned_model_path', required=True, type=str,
                       help='Path to model trained on clean + poison data')
    temp_args, remaining = parser.parse_known_args()
    args.poison_data_path = temp_args.poison_data_path
    args.poisoned_model_path = temp_args.poisoned_model_path

# Set default camouflage options
if args.camouflage_eps is None:
    args.camouflage_eps = args.eps
if args.camouflage_init is None:
    args.camouflage_init = args.init
if args.camouflage_restarts is None:
    args.camouflage_restarts = args.restarts
if args.camouflage_iter is None:
    args.camouflage_iter = args.attackiter

# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()


def load_poison_data(path):
    """Load existing poison data."""
    if path.endswith('.pth'):
        data = torch.load(path, map_location='cpu')
        if isinstance(data, list) and len(data) == 2:
            # Simple format: [poison_delta, poison_ids]
            poison_delta, poison_ids = data
            return poison_delta, poison_ids
        elif isinstance(data, dict):
            # Full format with setup info
            return data['poison_delta'], data['poison_ids']
    else:
        raise ValueError(f'Unsupported poison data format: {path}')


def load_model_state(model, filepath):
    """Load model state dict from file."""
    state_dict = torch.load(filepath, map_location='cpu')
    if hasattr(model, 'module'):  # Handle DataParallel
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    print(f'Model loaded from {filepath}')


if __name__ == "__main__":
    setup = forest.utils.system_startup(args)
    start_time = time.time()
    
    print("="*80)
    print("CAMOUFLAGE GENERATION FROM EXISTING POISON DATA")
    print("="*80)
    
    # Load existing poison data
    print(f"\nLoading poison data from {args.poison_data_path}...")
    try:
        poison_delta, poison_ids = load_poison_data(args.poison_data_path)
        print(f'Loaded {len(poison_ids)} poison samples')
    except Exception as e:
        print(f'Error loading poison data: {e}')
        exit(1)
    
    # Setup data and model infrastructure
    print("\nSetting up data and model infrastructure...")
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    
    # Load the poisoned model
    print(f"\nLoading poisoned model from {args.poisoned_model_path}...")
    try:
        poisoned_model = forest.Victim(args, setup=setup)
        poisoned_model.initialize(seed=model.model_init_seed)
        load_model_state(poisoned_model.model, args.poisoned_model_path)
    except Exception as e:
        print(f'Error loading poisoned model: {e}')
        exit(1)
    
    # Setup camouflage data
    print(f"\nSetting up camouflage data (budget: {args.camouflage_budget})...")
    data.setup_camouflage(camouflage_budget=args.camouflage_budget, 
                         camouflage_key=args.camouflage_key)
    
    if len(data.camouflage_ids) == 0:
        print('No camouflage samples available. Exiting.')
        exit(1)
    
    camouflage_setup_time = time.time()
    
    # Generate camouflage
    print(f"\nGenerating camouflage with {len(data.camouflage_ids)} samples...")
    print(f"Camouflage parameters:")
    print(f"  - eps: {args.camouflage_eps}")
    print(f"  - restarts: {args.camouflage_restarts}")
    print(f"  - iterations: {args.camouflage_iter}")
    print(f"  - initialization: {args.camouflage_init}")
    
    # Create camouflage witch
    camouflage_witch = forest.CamouflageWitch(args, setup, poisoned_model.model)
    
    # Generate camouflage
    camouflage_delta = camouflage_witch.brew(poisoned_model, data)
    
    generation_time = time.time()
    
    # Export camouflage
    print(f"\nExporting camouflage data to {args.poison_path}...")
    data.export_camouflage(camouflage_delta, path=args.poison_path, mode='packed')
    
    # Save camouflage in simple format as well
    simple_camouflage_path = os.path.join(args.poison_path, f'camouflage_simple_{datetime.date.today()}.pth')
    torch.save([camouflage_delta, data.camouflage_ids], simple_camouflage_path)
    print(f'Camouflage also saved in simple format to {simple_camouflage_path}')
    
    end_time = time.time()
    
    # Timing summary
    setup_time = camouflage_setup_time - start_time
    gen_time = generation_time - camouflage_setup_time  
    export_time = end_time - generation_time
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print("CAMOUFLAGE GENERATION COMPLETE")
    print("="*80)
    print(f'Setup time: {str(datetime.timedelta(seconds=setup_time))}')
    print(f'Generation time: {str(datetime.timedelta(seconds=gen_time))}')
    print(f'Export time: {str(datetime.timedelta(seconds=export_time))}')
    print(f'Total time: {str(datetime.timedelta(seconds=total_time))}')
    print(f'Optimal camouflage loss: {camouflage_witch.stat_optimal_loss:6.4e}')
    print(f'Generated {len(data.camouflage_ids)} camouflage samples')
    
    print(f"\nOutput files:")
    print(f'- Packed camouflage: {args.poison_path}/camouflages_packed_*.pth')
    print(f'- Simple camouflage: {simple_camouflage_path}')
    
    print("\n" + "="*80)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-------------Camouflage generation finished.-------------------------')
