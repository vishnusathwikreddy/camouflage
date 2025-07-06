"""Complete workflow script for poison + camouflage generation."""

import torch
import datetime
import time
import copy
import os

import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()

# Set default camouflage options if not specified
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


def save_model_state(model, path, name):
    """Save model state dict to file."""
    print(f"DEBUG: save_model_state called with path={path}, name={name}")
    print(f"DEBUG: model type: {type(model)}")
    
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f'{name}.pth')
    
    if hasattr(model, 'module'):  # Handle DataParallel
        state_dict = model.module.state_dict()
        print(f"DEBUG: Using DataParallel model, state_dict keys: {len(state_dict)}")
    else:
        state_dict = model.state_dict()
        print(f"DEBUG: Using regular model, state_dict keys: {len(state_dict)}")
    
    print(f"DEBUG: state_dict type: {type(state_dict)}")
    print(f"DEBUG: state_dict size: {len(state_dict) if state_dict else 'None'}")
    
    # Check if state_dict has content
    if state_dict:
        first_key = next(iter(state_dict))
        first_tensor = state_dict[first_key]
        print(f"DEBUG: First parameter '{first_key}' shape: {first_tensor.shape}")
        print(f"DEBUG: First parameter size: {first_tensor.numel()} elements")
    
    try:
        print(f"DEBUG: About to save to {filepath}")
        torch.save(state_dict, filepath)
        
        # Verify the save worked
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"DEBUG: Model file created: {filepath}, size: {file_size} bytes")
            
            # Try to load it back to verify it's valid
            try:
                test_load = torch.load(filepath, map_location='cpu')
                print(f"DEBUG: Model file verification: loaded {len(test_load)} keys successfully")
            except Exception as e:
                print(f"ERROR: Model file verification failed: {e}")
        else:
            print(f"ERROR: Model file was not created: {filepath}")
            
        print(f'Model saved to {filepath}')
        return filepath
        
    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")
        print(f"ERROR: Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None


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
    print("STARTING COMPLETE POISON + CAMOUFLAGE GENERATION WORKFLOW")
    print("="*80)
    
    # Step 1: Clean Model Training
    print("\n" + "="*60)
    print("STEP 1: CLEAN MODEL TRAINING")
    print("="*60)
    
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    
    if args.pretrained:
        print('Using pretrained model...')
        stats_clean = None
    else:
        print('Training clean model from scratch...')
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    
    # Save clean model
    clean_model_path = save_model_state(model.model, args.poison_path, 'clean_model')
    clean_train_time = time.time()
    
    # Step 2: Poison Generation
    print("\n" + "="*60)
    print("STEP 2: POISON GENERATION")
    print("="*60)
    
    witch = forest.Witch(args, setup=setup)
    poison_delta = witch.brew(model, data)
    
    # Export poison data - always export in packed mode for camouflage workflow
    print(f"DEBUG: About to export poison data...")
    print(f"DEBUG: args.save = {args.save}")
    print(f"DEBUG: args.poison_path = {args.poison_path}")
    print(f"DEBUG: poison_delta shape = {poison_delta.shape if poison_delta is not None else 'None'}")
    print(f"DEBUG: poison_delta type = {type(poison_delta)}")
    print(f"DEBUG: len(data.poison_ids) = {len(data.poison_ids) if hasattr(data, 'poison_ids') else 'No poison_ids'}")
    
    # Ensure poison_path directory exists
    if not os.path.exists(args.poison_path):
        os.makedirs(args.poison_path, exist_ok=True)
        print(f"DEBUG: Created directory {args.poison_path}")
    
    if args.save is not None:
        print(f"DEBUG: Calling export_poison with mode={args.save}")
        data.export_poison(poison_delta, path=args.poison_path, mode=args.save)
    else:
        # Always export in packed mode for camouflage workflow
        print("DEBUG: args.save is None, exporting in packed mode")
        data.export_poison(poison_delta, path=args.poison_path, mode='packed')
    
    poison_time = time.time()
    
    # Step 3: Train Model on Clean + Poison Data (for camouflage generation)
    print("\n" + "="*60)
    print("STEP 3: TRAINING MODEL ON CLEAN + POISON DATA")
    print("="*60)
    
    print(f"DEBUG: args.enable_camouflage = {getattr(args, 'enable_camouflage', 'NOT SET')}")
    
    if getattr(args, 'enable_camouflage', False):
        # Create a new model instance for poisoned training
        poisoned_model = forest.Victim(args, setup=setup)
        
        # Initialize with same seed as original for consistency
        poisoned_model.initialize(seed=model.model_init_seed)
        
        print('Training model on clean + poison data for camouflage generation...')
        stats_poisoned = poisoned_model.retrain(data, poison_delta)
        
        # Save poisoned model
        poisoned_model_path = save_model_state(poisoned_model.model, args.poison_path, 'poisoned_model')
        poisoned_train_time = time.time()
        
        # Step 4: Camouflage Generation
        print("\n" + "="*60)
        print("STEP 4: CAMOUFLAGE GENERATION")
        print("="*60)
        
        # Setup camouflage data
        print("DEBUG: Setting up camouflage data...")
        print(f"DEBUG: camouflage_budget = {args.camouflage_budget}")
        print(f"DEBUG: camouflage_key = {args.camouflage_key}")
        
        data.setup_camouflage(camouflage_budget=args.camouflage_budget, 
                             camouflage_key=args.camouflage_key)
        
        print(f"DEBUG: Camouflage setup result - IDs length: {len(data.camouflage_ids) if hasattr(data, 'camouflage_ids') and data.camouflage_ids is not None else 'None'}")
        
        if hasattr(data, 'camouflage_ids') and data.camouflage_ids is not None and len(data.camouflage_ids) > 0:
            # Create camouflage arguments (copy poison args but modify for camouflage)
            camouflage_args = copy.deepcopy(args)
            camouflage_args.eps = args.camouflage_eps
            camouflage_args.init = args.camouflage_init
            camouflage_args.restarts = args.camouflage_restarts
            camouflage_args.attackiter = args.camouflage_iter
            
            # Create camouflage witch with poisoned model
            camouflage_witch = forest.CamouflageWitch(camouflage_args, setup, poisoned_model.model)
            
            print(f'Generating camouflage with {len(data.camouflage_ids)} samples...')
            
            print("DEBUG: About to call camouflage_witch.brew()")
            print(f"DEBUG: poisoned_model type: {type(poisoned_model)}")
            print(f"DEBUG: data type: {type(data)}")
            
            try:
                camouflage_delta = camouflage_witch.brew(poisoned_model, data)
                print(f"DEBUG: Camouflage generation successful, delta shape: {camouflage_delta.shape}")
            except Exception as e:
                print(f"ERROR: Camouflage generation failed: {e}")
                print("DEBUG: Falling back to zero camouflage")
                camouflage_delta = torch.zeros(0, *data.trainset[0][0].shape)
            
            # Export camouflage data
            if len(camouflage_delta) > 0:
                data.export_camouflage(camouflage_delta, path=args.poison_path, mode='packed')
                print(f"DEBUG: Camouflage data exported to {args.poison_path}")
            else:
                print("DEBUG: No camouflage data to export")
            
            camouflage_time = time.time()
        else:
            print('No camouflage samples available - skipping camouflage generation')
            print(f"DEBUG: camouflage_ids exists: {hasattr(data, 'camouflage_ids')}")
            if hasattr(data, 'camouflage_ids'):
                print(f"DEBUG: camouflage_ids value: {data.camouflage_ids}")
            camouflage_delta = torch.zeros(0, *data.trainset[0][0].shape)
            camouflage_time = time.time()
        
        # Step 5: Final Model Training on Clean + Poison + Camouflage
        print("\n" + "="*60)
        print("STEP 5: FINAL MODEL TRAINING ON CLEAN + POISON + CAMOUFLAGE")
        print("="*60)
        
        # Create final model instance
        final_model = forest.Victim(args, setup=setup)
        final_model.initialize(seed=model.model_init_seed)
        
        print('Training final model on clean + poison + camouflage data...')
        
        # Debug: Check all data before training
        print(f"DEBUG: poison_delta shape: {poison_delta.shape}")
        print(f"DEBUG: camouflage_delta shape: {camouflage_delta.shape}")
        print(f"DEBUG: poison_ids length: {len(data.poison_ids)}")
        print(f"DEBUG: camouflage_ids exists: {hasattr(data, 'camouflage_ids')}")
        if hasattr(data, 'camouflage_ids') and data.camouflage_ids is not None:
            print(f"DEBUG: camouflage_ids length: {len(data.camouflage_ids)}")
        
        # Create combined dataset
        try:
            combined_dataset, combined_loader = data.get_combined_dataset(poison_delta, camouflage_delta)
            print("DEBUG: Combined dataset created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create combined dataset: {e}")
            print("DEBUG: Proceeding without combined dataset")
        
        # Debug: Check camouflage data
        if hasattr(data, 'camouflage_ids') and data.camouflage_ids is not None and len(data.camouflage_ids) > 0:
            print(f'Using {len(data.camouflage_ids)} camouflage samples with {len(camouflage_delta)} perturbations')
            if len(data.camouflage_ids) != len(camouflage_delta):
                print(f"WARNING: Mismatch between camouflage IDs ({len(data.camouflage_ids)}) and deltas ({len(camouflage_delta)})")
        else:
            print('No camouflage data available for final training')
        
        # Train on combined data using the new retrain_combined method
        try:
            print("DEBUG: About to call retrain_combined")
            stats_final = final_model.retrain_combined(data, poison_delta, camouflage_delta)
            print("DEBUG: retrain_combined completed successfully")
        except Exception as e:
            print(f"ERROR: retrain_combined failed: {e}")
            print("DEBUG: Falling back to regular retrain with poison only")
            stats_final = final_model.retrain(data, poison_delta)
        
        # Save final model
        final_model_path = save_model_state(final_model.model, args.poison_path, 'final_model')
        final_train_time = time.time()
        
        # Use final model for validation
        validation_model = final_model
        
    else:
        print("\nCamouflage generation disabled. Proceeding with standard poison workflow...")
        print("DEBUG: Setting camouflage_delta to empty tensor")
        camouflage_delta = torch.zeros(0, *data.trainset[0][0].shape)
        stats_poisoned = None
        stats_final = None
        poisoned_train_time = clean_train_time
        camouflage_time = poison_time
        final_train_time = poison_time
        validation_model = model
    
    # Ensure camouflage_delta is always defined
    if 'camouflage_delta' not in locals():
        print("DEBUG: camouflage_delta not defined, creating empty tensor")
        camouflage_delta = torch.zeros(0, *data.trainset[0][0].shape)
    
    print(f"DEBUG: Final camouflage_delta shape: {camouflage_delta.shape}")
    
    # Set default values for removed validation step
    stats_rerun = None
    stats_results = None
    
    end_time = time.time()
    
    # Prepare timing information
    timestamps = dict(
        clean_train_time=str(datetime.timedelta(seconds=clean_train_time - start_time)).replace(',', ''),
        poison_time=str(datetime.timedelta(seconds=poison_time - clean_train_time)).replace(',', ''),
        poisoned_train_time=str(datetime.timedelta(seconds=poisoned_train_time - poison_time)).replace(',', '') if args.enable_camouflage else 'N/A',
        camouflage_time=str(datetime.timedelta(seconds=camouflage_time - poisoned_train_time)).replace(',', '') if args.enable_camouflage else 'N/A',
        final_train_time=str(datetime.timedelta(seconds=final_train_time - camouflage_time)).replace(',', '') if args.enable_camouflage else 'N/A',
        total_time=str(datetime.timedelta(seconds=end_time - start_time)).replace(',', '')
    )
    
    # Record results
    results = (stats_clean, stats_rerun, stats_results)
    if args.enable_camouflage:
        # Save extended results separately
        extended_results = {
            'standard_results': results,
            'poisoned_model_stats': stats_poisoned,
            'final_model_stats': stats_final,
            'camouflage_info': {
                'num_camouflage_samples': len(data.camouflage_ids) if hasattr(data, "camouflage_ids") and data.camouflage_ids is not None else 0,
                'camouflage_budget': args.camouflage_budget,
                'camouflage_eps': args.camouflage_eps,
            }
        }
        # Save extended results to a separate file
        import pickle
        extended_path = os.path.join(args.poison_path, f'extended_results_{datetime.date.today()}.pkl')
        with open(extended_path, 'wb') as f:
            pickle.dump(extended_results, f)
        print(f'Extended results saved to {extended_path}')
        
        # Use standard results for the original record_results function
        results = (stats_clean, stats_rerun, stats_results)
    
    forest.utils.record_results(data, witch.stat_optimal_loss, results,
                               args, validation_model.defs, validation_model.model_init_seed, 
                               extra_stats=timestamps)
    
    # Final Summary
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*80)
    print(f'Clean model training time: {timestamps["clean_train_time"]}')
    print(f'Poison generation time: {timestamps["poison_time"]}')
    if args.enable_camouflage:
        print(f'Poisoned model training time: {timestamps["poisoned_train_time"]}')
        print(f'Camouflage generation time: {timestamps["camouflage_time"]}')
        print(f'Final model training time: {timestamps["final_train_time"]}')
        print(f'Number of camouflage samples: {len(data.camouflage_ids) if hasattr(data, "camouflage_ids") and data.camouflage_ids is not None else 0}')
    print(f'Total workflow time: {timestamps["total_time"]}')
    print(f'Optimal poison loss: {witch.stat_optimal_loss:6.4e}')
    
    print("\nGenerated files:")
    print(f'- Poison data: {args.poison_path}')
    if args.enable_camouflage and hasattr(data, "camouflage_ids") and data.camouflage_ids is not None and len(data.camouflage_ids) > 0:
        print(f'- Camouflage data: {args.poison_path}')
    print(f'- Model states: {args.poison_path}')
    
    print("\n" + "="*80)
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('-------------Camouflage workflow finished.-------------------------')
