import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import LinearSegmentedColormap
import os
import fnmatch
from omegaconf import DictConfig

def log_heatmaps(targets, preds, n_examples=20, n_res=4, cmap='coolwarm'):
    # log qualitative results
    targets_case = targets.detach().cpu().numpy().squeeze().reshape(-1, n_res, 512, 512)
    preds_case = preds.detach().cpu().numpy().squeeze().reshape(-1, n_res, 512, 512)
    for i in range(n_examples):
        fig, axes = plt.subplots(n_res, 2, figsize=(10, 16))
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        for resistancy in range(n_res):
            axes[resistancy,0].imshow(targets_case[i,resistancy], vmin=0, vmax=0.7, cmap=cmap)
            axes[resistancy,0].axis('off')
            axes[resistancy,1].imshow(preds_case[i,resistancy], vmin=0, vmax=0.7, cmap=cmap)
            axes[resistancy,1].axis('off')
        # Add colorbar to the figure
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(0, 0.7)
        cbar = fig.colorbar(sm, cax=cbar_ax)        
        cbar.set_label('Specific Conductivity (S/m)')
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

def get_all_cases(cfg: DictConfig, base_dir=".."):
    if cfg.data.cases == 'all':
        cases = os.listdir(os.path.join(base_dir,cfg.data.processed_data_folder))
        cases = [case.split('.')[0] for case in cases if fnmatch.fnmatch(case, 'case_TCIA*')]
        cases_number = [int(case.split('_')[-2]) for case in cases]
        # cases = [case for case, case_number in zip(cases, cases_number) if case_number < 290]
        # cases 
    else:
        cases = cfg.data.cases
    return cases
