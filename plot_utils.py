import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import torch # Needed to load tensors from pickle

def plot_pickle(fname: str):
    """Load a pickled results dict and produce two pcolormesh figures."""
    try:
        with open(fname, 'rb') as f:
            ret = pickle.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Pickle file not found: {fname}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file {fname}: {e}")
        return

    # Extract data, converting to numpy and ensuring correct types
    try:
        # Args might be a class instance or dict, handle accordingly
        args_data = ret.get('args', None)
        if args_data is None:
             print("[WARN] 'args' not found in pickle file. Cannot generate descriptive plot filename.")
             # Use original pickle filename as fallback base for plot names
             args_available = False
             # Provide dummy args or defaults if needed for plotting logic below
             class DummyArgs:
                 plotdim = False
                 step = 0.1
                 use_softmax = True # Assume default if missing
                 dmodel = 'N/A'
                 ntokens = 'N/A'
                 norm = 'N/A' # Add norm for consistency
             args = DummyArgs()
        else:
            args_available = True
            # If args is a dict (saved as vars(args)), convert to Namespace or object for attribute access
            if isinstance(args_data, dict):
                 from argparse import Namespace
                 args = Namespace(**args_data)
            else:
                 args = args_data # Assume it's already an object (like Namespace or custom class)

        betas_list = ret.get('betas_list', None)
        times_list = ret.get('times_list', None)
        density_tensor = ret.get('density_tensor', None)
        # cluster_tensor = ret.get('cluster_tensor', None) # Currently unused in plot logic

        if any(x is None for x in [betas_list, times_list, density_tensor]):
             print("[ERROR] Missing essential data (betas, times, density) in pickle file.")
             return

        # Ensure tensors are numpy arrays on CPU
        betas_list = betas_list.cpu().numpy() if isinstance(betas_list, torch.Tensor) else np.array(betas_list)
        times_list = times_list.cpu().numpy() if isinstance(times_list, torch.Tensor) else np.array(times_list)
        density_tensor = density_tensor.cpu().numpy() if isinstance(density_tensor, torch.Tensor) else np.array(density_tensor)

        if density_tensor.shape != (len(betas_list), len(times_list)):
             print(f"[ERROR] Shape mismatch: density {density_tensor.shape}, betas {betas_list.shape}, times {times_list.shape}")
             return

    except Exception as e:
        print(f"[ERROR] Failed to extract or process data from pickle: {e}")
        return

    # Create meshgrid for plotting
    X, Y = np.meshgrid(times_list, betas_list) # meshgrid expects (x_coords, y_coords)

    # Determine plot titles and settings from args
    # Safely access attributes from args
    plotdim = getattr(args, 'plotdim', False)
    step = getattr(args, 'step', 0.1)
    use_softmax = getattr(args, 'use_softmax', True)
    dmodel = getattr(args, 'dmodel', 'N/A')
    ntokens = getattr(args, 'ntokens', 'N/A')
    norm_val = getattr(args, 'norm', 'N/A') # Use norm_val to avoid conflict with matplotlib.colors.Normalize
    
    setup = f"Step={step}, Softmax={'Yes' if use_softmax else 'No'}"
    title_base = f"D={dmodel}, N={ntokens}, Norm={norm_val}, {setup}"

    # --- Determine Output Path and Base Filename ---
    output_dir = os.path.dirname(fname) # Assume plots go in the same dir as the pickle
    if not output_dir:
        output_dir = "." # Default to current directory if path is just a filename
    
    if args_available:
        # Construct a descriptive base filename
        base_plot_fname = f"plot_d{dmodel}_n{ntokens}_norm{norm_val}"
    else:
        # Fallback to using the pickle filename base if args weren't available
        base_plot_fname = os.path.splitext(os.path.basename(fname))[0]
    # --- 

    # Plotting loops (Linear and Log scale for inverse density, or dim plot)
    if plotdim:
        # Special case for dimension plot
        plot_configs = [
            {'suffix': '_dim.png', 'cmap': 'RdBu_r', 'norm': None, 
             'title': f"Density Dimension - {title_base}", 'data': density_tensor, 
             'cbar_label': 'Dimension'}
        ]
    else:
        # Inverse density plots (linear and log)
        inverse_density = 1.0 / np.clip(density_tensor, 1e-6, None)
        inverse_density_clipped = np.clip(inverse_density, None, 10)
        plot_configs = [
            {'suffix': '_invdens_linear.png', 'cmap': 'viridis', 
             'norm': Normalize(vmin=0, vmax=10), 
             'title': f"Inverse Density (Linear) - {title_base}", 
             'data': inverse_density_clipped, 'cbar_label': 'Inv. Density (Clipped)', 
             'ticks': np.arange(0, 11, 1)},
            {'suffix': '_invdens_log.png', 'cmap': 'viridis', 
             'norm': LogNorm(vmin=1e-6, vmax=10), # Ensure vmin > 0 for LogNorm
             'title': f"Inverse Density (Log) - {title_base}", 
             'data': inverse_density_clipped, 'cbar_label': 'Inv. Density (Clipped)', 
             'ticks': None}
        ]

    # Common plot settings
    ylabel = "Beta"
    xlabel = "Time"

    for config in plot_configs:
        fig, ax = plt.subplots()
        
        pc = ax.pcolormesh(X, Y, config['data'], cmap=config['cmap'], norm=config['norm'], shading='auto')

        # Add the colorbar
        cbar = fig.colorbar(pc, ax=ax, label=config['cbar_label'])
        
        # Set colorbar ticks if specified (e.g., for linear scale)
        if config.get('ticks', None) is not None:
             cbar.set_ticks(config['ticks'])
             cbar.set_ticklabels([str(t) for t in config['ticks']])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(config['title'])

        # Construct final plot path
        fig_path = os.path.join(output_dir, f"{base_plot_fname}{config['suffix']}")
        
        try:
            fig.savefig(fig_path)
            print(f"Saved plot: {fig_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save plot {fig_path}: {e}")
        
        plt.close(fig) # Close figure to free memory 