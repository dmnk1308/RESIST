import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import LinearSegmentedColormap
import os
import fnmatch
import random
import torch
import numpy as np
from omegaconf import DictConfig

def log_heatmaps(targets, preds, n_examples=20, n_levels=4, cmap='coolwarm', resolution=512):
    # log qualitative results
    targets_case = targets.detach().cpu().numpy().squeeze().reshape(-1, n_levels, resolution, resolution)
    preds_case = preds.detach().cpu().numpy().squeeze().reshape(-1, n_levels, resolution, resolution)
    for i in range(n_examples):
        fig, axes = plt.subplots(n_levels, 2, figsize=(10, 16))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        for level in range(n_levels):
            axes[level,0].imshow(targets_case[i,level], vmin=0, vmax=0.7, cmap=cmap)
            axes[level,0].axis('off')
            axes[level,1].imshow(preds_case[i,level], vmin=0, vmax=0.7, cmap=cmap)
            axes[level,1].axis('off')
        # Add colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax)        
        cbar.set_label('Conductivity (S/m)')
        wandb.log({'Heatmap': fig})
        plt.close(fig)

def make_cmap():
    # Define the colors and their positions in the colormap
    colors = [(1., 1., 1),   # white
            (0.9, 0.9, 0.9),
            (0.1, 0.1, 1),   # light blue
            (1, 0.8, 0.4),   # yellow
            (0.8, 0, 0)]     # red
    positions = [0, 0.01, 0.05, 0.2, 1.0]

    # Create the colormap
    cmap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)))
    return cmap

def get_all_cases(cfg: DictConfig, base_dir="..", use_raw=False):
    if cfg.data.cases == 'all':
        if use_raw:
            cases = os.listdir(os.path.join(base_dir,cfg.data.raw_data_folder))
        else:
            cases = os.listdir(os.path.join(base_dir,cfg.data.processed_data_folder))
        cases = [case.split('.')[0] for case in cases if fnmatch.fnmatch(case, 'case_TCIA*')]
        # cases = [case for case in cases if os.path.exists(os.path.join(base_dir,cfg.data.raw_data_folder,case,'shape','mesh.vtk'))]
        # cases = [case for case in cases if not os.listdir(os.path.join(base_dir,cfg.data.raw_data_folder,case,'shape'))]
        cases_number = [int(case.split('_')[-2]) for case in cases]
        # cases = [case for case, case_number in zip(cases, cases_number) if (case_number!=250)]
        cases_number = [case_number for case_number in cases_number if case_number!=250]
        cases_number.sort()
        cases = ['case_TCIA_'+str(case_number)+'_0' for case_number in cases_number]
        # idx = np.argsort(np.array(cases_number))
        # cases = np.array(cases)[idx]
        # cases = cases.tolist()
    else:
        cases = cfg.data.cases
    return cases

def set_seeds(seed=123):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


            