import argparse
import torch
import numpy as np
import pickle
import os
import time
import gc
import math
import tqdm

from simuxform import SimuXForm
from plot_utils import plot_pickle

# Default Hyperparameters (can be overridden by CLI args where applicable)
N_HEADS = 8
BATCH_SIZE = 512
BETAMIN = 0.1
BETAMAX = 15.0
N_BETAS = 20
MAXTIME = 50.0 # Default simulation time
STEP_SIZE = 0.1
N_TOKENS = 64
D_MODEL = 16
DEFAULT_NORM = "preln"
DEFAULT_RANDOM_V = 0
DEFAULT_RANDOM_KQ = 1
DEFAULT_N_TIME_CHECKPOINTS = 200 # Number of time points to record stats

# --- Device Selection Logic ---
def get_compute_device():
    """Selects the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        print("[INFO] CUDA is available. Using CUDA.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[INFO] MPS is available. Using MPS.")
        return torch.device("mps")
    else:
        print("[INFO] CUDA and MPS not available. Using CPU.")
        return torch.device("cpu")

DEVICE = get_compute_device()
# ---

def parse_args():
    parser = argparse.ArgumentParser(description="Run SimuXForm particle simulation experiments.")
    
    # Core parameters from spec
    parser.add_argument("--norm", type=str, default=DEFAULT_NORM, choices=["postln", "preln", "periln", "ngpt"], 
                        help="Normalization mode (postln, preln, periln, or ngpt)")
    parser.add_argument("--randomV", type=int, default=DEFAULT_RANDOM_V, choices=[0, 1, 2, 3, 4],
                        help="V matrix initialization mode (0=identity, 1=random, 2=BF/norm, 3=-BF/norm, 4=GPT-2 style std=0.02)")
    parser.add_argument("--randomKQ", type=int, default=DEFAULT_RANDOM_KQ, choices=[0, 1, 2, 3, 4],
                        help="KQ bilinear form init mode (0=identity, 1=random*random, 2=symmetric, 3=orthogonal-like, 4=GPT-2 style std=0.02)")
    parser.add_argument("--sequential", action="store_true", default=False,
                        help="Use sequential (causal) masking in attention")

    # Hyperparameters that might be useful as CLI args
    parser.add_argument("--dmodel", type=int, default=D_MODEL, help="Model dimension")
    parser.add_argument("--ntokens", type=int, default=N_TOKENS, help="Number of tokens (particles)")
    parser.add_argument("--n_heads", type=int, default=N_HEADS, help="Number of attention heads")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--betamin", type=float, default=BETAMIN, help="Minimum beta value")
    parser.add_argument("--betamax", type=float, default=BETAMAX, help="Maximum beta value")
    parser.add_argument("--betas", type=int, default=N_BETAS, help="Number of beta values to sweep")
    parser.add_argument("--maxtime", type=float, default=MAXTIME,
                        help="Maximum simulation time (default adjusts based on softmax usage if 0.0)")
    parser.add_argument("--step", type=float, default=STEP_SIZE, help="Simulation time step size")
    parser.add_argument("--n_time_ckpts", type=int, default=DEFAULT_N_TIME_CHECKPOINTS,
                        help="Number of time checkpoints to record data")


    # Extra parameters found in original code (kept for now)
    parser.add_argument("--adjacent", action="store_true", default=False,
                        help="Calculate density based on adjacent tokens only")
    parser.add_argument("--cluster_sizes", action="store_true", default=False,
                        help="Calculate median cluster size metric")
    parser.add_argument("--disable_tqdm", action="store_true", default=False,
                        help="Disable tqdm progress bar")
    parser.add_argument("--noanneal", type=int, default=0, choices=[0, 1, 2],
                        help="Annealing mode for K/Q/V matrices (0: regen each beta, 1: shared across batch, 2: reuse first)")
    parser.add_argument("--rawstep", action="store_true", default=False,
                        help="Use raw step size eta, ignoring beta-based correction")
    # parser.add_argument("--regen_period", type=float, default=0.0, help="Regeneration period (unused?)") # Seems unused
    parser.add_argument("--plotdim", action="store_true", default=False,
                        help="Calculate density based on eigenvalue threshold")
    parser.add_argument("--analyze_evecs", action="store_true", default=False,
                        help="Analyze and print V matrix eigenvector similarity")
    parser.add_argument("--use_softmax", action="store_true", default=True,
                        help="Use softmax for attention normalization (alternative is scaled exp)")
    parser.add_argument("--no_softmax", action="store_false", dest="use_softmax",
                         help="Use scaled exp for attention normalization instead of softmax")

    # Output control
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--no_plot", action="store_true", default=False, help="Skip plotting results after simulation")

    args = parser.parse_args()
    
    # Add the determined device to args for easy access later
    args.device = DEVICE 
    
    return args

def do_single_beta(args, beta_idx, ret_dict, V_shared=None, BF_shared=None):
    """Runs the simulation for a single beta value."""
    # Ensure device is accessible from args
    device = args.device 
    beta = ret_dict['betas_list_device'][beta_idx].to(device) # Ensure beta is on the correct device
    
    # Pass shared V/BF if provided (for noanneal=2)
    s = SimuXForm(args, beta, V=V_shared, BF=BF_shared)

    cur_ckpt = 0
    cur_time = 0.0
    # Use device tensors for accumulation
    dt = ret_dict['density_tensor_device'] 
    ct = ret_dict['cluster_tensor_device']
    times_list_device = ret_dict['times_list_device']
    max_ckpt = len(times_list_device)

    # Simulation loop for this beta
    while True:
        cur_time = s.step(cur_time)

        # Record statistics if current time passes the next checkpoint time
        if cur_ckpt < max_ckpt and cur_time >= times_list_device[cur_ckpt]:
            # Perform check_stats on the correct device
            dens, clust = s.check_stats()
            # Fill potentially multiple checkpoints if time step is large
            while cur_ckpt < max_ckpt and cur_time >= times_list_device[cur_ckpt]:
                dt[beta_idx, cur_ckpt] = dens
                ct[beta_idx, cur_ckpt] = clust
                cur_ckpt += 1

        # Exit conditions
        if cur_ckpt >= max_ckpt or cur_time >= args.maxtime:
            # Ensure final state is recorded if needed
            if cur_ckpt < max_ckpt:
                 dens, clust = s.check_stats()
                 dt[beta_idx, cur_ckpt:] = dens # Fill remaining checkpoints with last value
                 ct[beta_idx, cur_ckpt:] = clust
            break

    # Clean up SimuXForm instance and potentially free memory
    del s
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # MPS cache emptying might differ or be less necessary
        # torch.mps.empty_cache() # Check if this function exists and is needed
        pass # Currently no explicit empty_cache for MPS in stable releases
    gc.collect()


def do_results(args):
    """Orchestrates the simulation sweep over beta values."""
    device = args.device # Use the device determined earlier
    torch.set_default_dtype(torch.float32)
    print(f'Using device: {device}')

    ret_dict = {}
    # Keep lists/tensors on device for computation
    betas_list_device = torch.linspace(args.betamin, args.betamax, args.betas, device=device)
    times_list_device = torch.linspace(0, args.maxtime, args.n_time_ckpts, device=device)
    
    ret_dict['betas_list_device'] = betas_list_device
    ret_dict['times_list_device'] = times_list_device
    # Create CPU copies *only* when needed for saving at the end
    # ret_dict['betas_list'] = betas_list_device.cpu() 
    # ret_dict['times_list'] = times_list_device.cpu()

    # Initialize result tensors on device
    density_tensor_device = torch.zeros(len(betas_list_device), len(times_list_device), device=device)
    cluster_tensor_device = torch.zeros(len(betas_list_device), len(times_list_device), device=device)
    ret_dict['density_tensor_device'] = density_tensor_device
    ret_dict['cluster_tensor_device'] = cluster_tensor_device

    V_shared = None
    BF_shared = None
    if args.noanneal >= 2:
        print("[INFO] Pre-generating shared K/Q/V matrices for noanneal=2")
        try:
             dummy_beta = torch.tensor(1.0, device=device) 
             s_init = SimuXForm(args, dummy_beta)
             # Ensure shared tensors are on the correct device
             V_shared = [v.clone().to(device) for v in s_init.V if v is not None] 
             BF_shared = [bf.clone().to(device) for bf in s_init.BF if bf is not None]
             del s_init
             if device.type == 'cuda':
                 torch.cuda.empty_cache()
             # elif device.type == 'mps': # No empty_cache needed/available
             #    pass 
             gc.collect()
             print("[INFO] Shared matrices generated and moved to device.")
        except Exception as e:
             print(f"[ERROR] Failed to pre-generate shared matrices: {e}. Proceeding without sharing.")
             V_shared = None
             BF_shared = None

    # Loop over beta values
    beta_iterable = enumerate(betas_list_device)
    if not args.disable_tqdm:
        beta_iterable = tqdm.tqdm(beta_iterable, total=len(betas_list_device), desc='Beta Sweep')

    for i, beta in beta_iterable:
        # Pass shared tensors which are already on the correct device
        do_single_beta(args, i, ret_dict, V_shared=V_shared, BF_shared=BF_shared)

    # Prepare results for saving (move to CPU)
    ret_dict['betas_list'] = ret_dict['betas_list_device'].cpu()
    ret_dict['times_list'] = ret_dict['times_list_device'].cpu()
    ret_dict['density_tensor'] = ret_dict['density_tensor_device'].cpu()
    ret_dict['cluster_tensor'] = ret_dict['cluster_tensor_device'].cpu()
    
    # Clean up device tensors from the dict
    del ret_dict['density_tensor_device']
    del ret_dict['cluster_tensor_device']
    del ret_dict['betas_list_device']
    del ret_dict['times_list_device']

    return ret_dict

if __name__ == "__main__":
    args = parse_args() # args now includes args.device
    device = args.device # Convenience variable

    # Maxtime adjustment logic (kept as is)
    if args.maxtime == MAXTIME:
        adjusted_maxtime = 30.0 if args.use_softmax else 1.0
        print(f"[INFO] Adjusting maxtime based on use_softmax: {adjusted_maxtime}")
        print(f"[INFO] Consider setting --maxtime explicitly. Using specified/default: {args.maxtime}")

    # Filename generation (kept as is)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fname_prefix = (
        f"d{args.dmodel}_n{args.ntokens}_nh{args.n_heads}_" 
        f"norm{args.norm}_V{args.randomV}_KQ{args.randomKQ}_seq{args.sequential}_" 
        f"beta{args.betas}x{args.betamin:.1f}-{args.betamax:.1f}_T{args.maxtime:.1f}_" 
        f"{timestamp}"
    )
    os.makedirs(args.outdir, exist_ok=True)
    result_basename = os.path.join(args.outdir, fname_prefix)
    result_filename = f"{result_basename}.pkl"
    log_filename = f"{result_basename}.log"

    # --- Check if result file already exists ---
    if os.path.exists(result_filename):
        print(f"\n[INFO] Result file already exists, skipping computation:")
        print(f"  {result_filename}")
        # Optionally, load existing results if needed later, though not strictly necessary
        # results = None # Or load pickle if you need args from it for plotting title fallback
    else:
        print(f'\n--- Starting Experiment ---')
        print(f'Device: {device}') # Print the selected device
        # Convert Namespace to dict for printing/saving if needed
        args_dict = vars(args)
        # Remove device object before printing/saving if it causes issues
        args_dict_printable = {k: v for k, v in args_dict.items() if k != 'device'}
        args_dict_printable['device_type'] = str(device.type) # Store device type as string
        print(f'Parameters: {args_dict_printable}') 
        print(f'Output prefix: {result_basename}')

        start_time = time.time()

        # Run the experiment sweep
        results = do_results(args=args)

        # Add args to the results dict for saving context
        # Save the printable version without the device object
        results['args'] = args_dict_printable 

        # Save results to a pickle file
        try:
            with open(result_filename, 'wb') as f:
                pickle.dump(results, f)
            print(f'Results saved to: {result_filename}')
        except Exception as e:
            print(f"[ERROR] Failed to save results pickle: {e}")

        end_time = time.time()
        total_time_sec = end_time - start_time
        total_time_min = total_time_sec / 60.0
        print(f'--- Experiment Completed ---')
        print(f'Total time: {total_time_sec:.2f} seconds ({total_time_min:.2f} minutes)')

        # Save log content to a file
        log_content = f'Experiment Settings:\n{args_dict_printable}\n\n'
        log_content += f'Device: {device.type}\n' # Use device.type
        log_content += f'Results File: {result_filename}\n'
        log_content += f'Total execution time: {total_time_sec:.2f} seconds ({total_time_min:.2f} minutes)\n'
        try:
            with open(log_filename, 'w') as f:
                f.write(log_content)
            print(f'Log saved to: {log_filename}')
        except Exception as e:
            print(f"[ERROR] Failed to save log file: {e}")

    # --- Plotting (runs even if computation was skipped) ---
    if not args.no_plot:
        # Check if the result file exists before trying to plot
        if os.path.exists(result_filename):
            print("\nGenerating plots...")
            # Pass the original args object (which includes device type string) to plot_pickle if needed
            # Currently plot_pickle reads args from the pickle file, which now contains the string 'device_type'
            # plot_pickle expects the filename, not the results dict
            plot_pickle(result_filename)
        else:
            print(f"\n[WARN] Cannot plot. Result file not found (and computation was skipped or failed): {result_filename}")
    else:
        print("\nPlotting skipped (--no_plot specified).")