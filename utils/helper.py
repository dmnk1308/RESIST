import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import LinearSegmentedColormap
import os
import fnmatch
import random
import torch
import numpy as np
from omegaconf import DictConfig
import re

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
        cases_number = [case_number for case_number in cases_number if case_number!=250]
        cases_number = [case_number for case_number in cases_number if case_number!=428]
        cases_number = [case_number for case_number in cases_number if case_number!=385]
        cases_number = [case_number for case_number in cases_number if case_number!=403]
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

def extract_rho_number(strings):
    pattern = re.compile(r'rho_(\d+)')
    extracted = [int(pattern.search(string).group(1)) if pattern.search(string) else None for string in strings]
    return extracted

def extract_level_number(strings, processed=False):
    if processed:
        pattern = re.compile(r'level_(\d+)_')
    else:
        pattern = re.compile(r'rho_\d+_z(\d+)')
    extracted = [int(pattern.search(string).group(1)) if pattern.search(string) else None for string in strings]
    return extracted

def sort_strings_by_numbers(strings, first_list, second_list):
    # Combine the strings with their corresponding numbers
    combined = list(zip(strings, first_list, second_list))
    
    # Sort the combined list by first_list and then by second_list
    sorted_combined = sorted(combined, key=lambda x: (x[1], x[2]))
    
    # Extract the sorted strings, first list, and second list
    sorted_strings = [item[0] for item in sorted_combined]
    sorted_first_list = [item[1] for item in sorted_combined]
    sorted_second_list = [item[2] for item in sorted_combined]
    
    return sorted_strings, sorted_first_list, sorted_second_list    

def get_rho_level_sorted_signals(dir):
    '''
    Takes a "signals" directory and returns the sorted path to each measurement, 
    the body level and the used lung rho as lists
    '''
    files = os.listdir(dir)
    files = [signal for signal in files if signal.endswith('.get')]
    rhos = extract_rho_number(files)
    files = [files[i] for i in range(len(files)) if rhos[i] is not None]
    rhos = [r for r in rhos if r is not None]
    levels = extract_level_number(files)
    files = [files[i] for i in range(len(files)) if levels[i] is not None]
    levels = [l for l in levels if levels is not None]
    files, levels, rhos = sort_strings_by_numbers(files, levels, rhos)
    files = [os.path.join(dir,f) for f in files]
    return files, levels, rhos

def move_to_level_summary(dir):
    '''
    Takes a "signals" directory and moves all files which have the pattern *_<rho1>_<rho2>_<level> 
    to a level summary directory and renames them accordingly
    '''
    strings = os.listdir(dir)
    level_summary_path = os.path.join(dir,'level_summary')
    os.makedirs(level_summary_path, exist_ok=True)
    pattern = re.compile(r'_(\d+)_(\d+)_z(\d+)')

    for string in strings:
        match = pattern.search(string)
        if match:
            number1 = match.group(1)
            number2 = match.group(2)
            number3 = match.group(3)
            dest_file_name = f"level_{number3}_rho_{number1}_to_{number2}.get"
            dest_file = os.path.join(level_summary_path,dest_file_name)
            source_file = os.path.join(dir, string)
            os.rename(source_file, dest_file)