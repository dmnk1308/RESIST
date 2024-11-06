import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import fnmatch
import torch
from train.testing import testing
from data_processing.obj2py import read_get
from data_processing.helper import combine_electrode_positions
from plotting_helper import *
import argparse

def reconstruct(data, 
                model_path=None, 
                electrodes=None, 
                points=None, 
                device='cuda', 
                resolution=512, 
                zpos=None, 
                n_zpos=4, 
                axis='axial',
                z_padding=0,
                multiplier=1000/4.5,
                verbose=True,
                load_std_files=True,
                return_aspect_ratio=False):
    # get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # load model and normalizing statistics
    if model_path is None:
        model_path = os.path.join(script_dir, 'outputs', 'resist')
        model_path = os.path.relpath(model_path,script_dir)
    model, cfg = load_model(model_path, device='cuda')

    try:
        signals_mean = torch.load(os.path.join(cfg.data.dataset_data_folder, f'train_dataset{cfg.data.name_prefix}.pt')).train_mean.numpy()
        signals_std = torch.load(os.path.join(cfg.data.dataset_data_folder, f'train_dataset{cfg.data.name_prefix}.pt')).train_std.numpy()
        points_max = torch.load(os.path.join(cfg.data.dataset_data_folder, f'train_dataset{cfg.data.name_prefix}.pt')).points_max.numpy()
        points_min = torch.load(os.path.join(cfg.data.dataset_data_folder, f'train_dataset{cfg.data.name_prefix}.pt')).points_min.numpy()
        if not os.path.exists('outputs/resist/signals_mean.npy'):
            np.save('outputs/resist/signals_mean.npy', signals_mean)
            np.save('outputs/resist/signals_std.npy', signals_std)
            np.save('outputs/resist/points_max.npy', points_max)
            np.save('outputs/resist/points_min.npy', points_min)
    except:
        print('Could not load normalizing statistics, loading from file instead.')
        signals_mean = np.load('outputs/resist/signals_mean.npy')
        signals_std = np.load('outputs/resist/signals_std.npy')
        points_max = np.load('outputs/resist/points_max.npy')
        points_min = np.load('outputs/resist/points_min.npy')

    # load signals
    if isinstance(data, np.ndarray):
        signals = data
    
    elif os.path.isdir(data):
        signals = []

        files = os.listdir(data)
        files.sort()
        files_signals = [os.path.join(data, f) for f in files if fnmatch.fnmatch(f, '*.npy')]
        if len(files_signals)==0:
            if verbose:
                print('No .npy files found, try .get files instead.')
            files_signals = [os.path.join(data, f) for f in files if fnmatch.fnmatch(f, '*.get')]
            for p in files_signals:
                signals.append(read_get(p)[:208])
        else:
            for p in files_signals:
                signals.append(np.load(p)[:208])
        if verbose:
            print('Using the following files and ordering:', files_signals)
        signals = np.stack(signals, axis=0)
    else:
        if verbose:
            print('Using the following single file:', data)
        signals = read_get(data)[:208]
    signals = signals.reshape(-1, 16, 13)*multiplier

    # load electrodes position if available, otherwise use default and interplolate number of levels
    if electrodes is None:
        if verbose:
            print('Using default electrode positions.')
        electrodes = np.load(os.path.join(parent_dir,'data/defaults/electrodes.npy'))
        electrodes = electrodes.reshape(4,-1,3)
        electrodes = np.linspace(electrodes[0], electrodes[-2], signals.shape[0])
    else:
        print(f'Use electrodes at {electrodes}.')
        electrodes = np.load(electrodes)
        electrodes = electrodes.reshape(-1, 16, 3)
        electrodes = electrodes - electrodes[0,0].reshape(1,1,3)
    if return_aspect_ratio:
        max, min = electrodes.reshape(-1,3).max(axis=0), electrodes.reshape(-1,3).min(axis=0)
        x_dist = max[0] - min[0]
        y_dist = max[1] - min[1]
        z_dist = max[2] - min[2]
        aspect_ratio_axial = y_dist / x_dist
        aspect_ratio_coronal = z_dist / x_dist
        aspect_ratio_sagittal = z_dist / y_dist
        aspect_ratio = [aspect_ratio_axial, aspect_ratio_coronal, aspect_ratio_sagittal]
    electrodes = electrodes.reshape(-1, 16, 3)
    signals_mean = signals_mean[:,:signals.shape[0]]
    signals_std = signals_std[:,:signals.shape[0]]

    # standardize
    if len(signals.shape) <= 3:
        signals = signals.reshape(-1, signals_mean.shape[1], signals_mean.shape[2], signals_mean.shape[3])
    if len(electrodes.shape) <= 3:
        electrodes = combine_electrode_positions(electrodes)
        electrodes = electrodes[None]
    else:
        e_tmp = []
        for i in range(electrodes.shape[0]):
            e_tmp.append(combine_electrode_positions(electrodes[i]))
        electrodes = np.stack(e_tmp, axis=0)
    batch_size = signals.shape[0]
    signals = ((signals - signals_mean) / signals_std).reshape(batch_size, signals.shape[1], -1)
    electrodes = ((electrodes - points_min) / (points_max - points_min)) * 2 - 1

    # set resolution for axes
    resolution_x = resolution
    resolution_y = resolution
    resolution_z = resolution
    if axis == 'axial':
        resolution_z = n_zpos
    elif axis == 'coronal':
        resolution_x = n_zpos
    elif axis == 'sagittal':
        resolution_y = n_zpos
    else:
        raise ValueError('axis must be one of axial, sagittal, or coronal')

    if zpos is None:
        max_z = np.max(electrodes.reshape(-1,3)[:,-1])
        min_z = np.min(electrodes.reshape(-1,3)[:,-1])
        zpos = np.linspace(max_z+z_padding, min_z, n_zpos)

    if points is None:
        # get max and min points from electrode positions
        max_x, max_y, max_z = np.max(electrodes.reshape(-1,3), axis=0)
        min_x, min_y, min_z = np.min(electrodes.reshape(-1,3), axis=0)
        x = np.linspace(min_x, max_x, resolution_x)
        y = np.linspace(min_y, max_y, resolution_y)
        z = np.linspace(max_z+z_padding, min_z, resolution_z)
        xyz = np.meshgrid(x, y, z)
        # transpose grid to (z, y, x, 3) and switch x and y
        xyz = np.array(xyz).T
        xyz = np.moveaxis(xyz,2,1).reshape(1, -1, 3)
        points = np.tile(xyz, (batch_size, 1, 1))
    

    if batch_size > 1:
        pred_tmp = []
        for i in tqdm(range(batch_size)):
            _, pred, _ = testing(model, [signals[i][None], electrodes[i][None], points[i][None]], device=device, wandb_log=False, point_levels_3d=len(zpos), point_chunks=len(zpos), resolution=resolution)
            pred_tmp.append(pred.cpu())
        pred = torch.stack(pred_tmp, dim=0)
    else:
        _, pred, _ = testing(model, [signals, electrodes, points], device=device, wandb_log=False, point_levels_3d=len(zpos), point_chunks=len(zpos), resolution=resolution)
    pred = pred.reshape(batch_size, resolution_z, resolution_y, resolution_x).numpy()
    # move view axis to front
    if axis == 'axial':
        pred = np.moveaxis(pred, 1, 1)
    elif axis == 'coronal':
        pred = np.moveaxis(pred, 2, 1)
    elif axis == 'sagittal':
        pred = np.moveaxis(pred, 3, 1)

    if return_aspect_ratio:
        return pred, aspect_ratio
    return pred


# real data
def plot_real_data(case_dir, rec_number, cut, electrodes, z_padding, resolution=512, slice_axial=16, slice_coronal=64, device='cuda', save=None):
    files = os.listdir(case_dir)
    files = [os.path.join(case_dir, f) for f in files if f.endswith('.get')]
    files = [file for file in files if fnmatch.fnmatch(file, f'*{str(rec_number).zfill(2)}_SF_*')]
    cmap = make_cmap()

    # loading data 
    data_max_levels = []
    data_min_levels = []
    for level in [1,2,3]:
        for file in files:
            if fnmatch.fnmatch(file, f'*SF_{level}_*_{cut}_max_mean.get'):
                data_max = read_get(file)[:208].reshape(-1,16,13)
                data_max_levels.append(data_max)
                print(file)
            if fnmatch.fnmatch(file, f'*SF_{level}_*_{cut}_min_mean.get'):
                data_min = read_get(file)[:208].reshape(-1,16,13)
                data_min_levels.append(data_min)
                print(file)
    data_max_levels = np.array(data_max_levels)
    data_min_levels = np.array(data_min_levels)

    # prediction
    pred_max, aspect_ratio = reconstruct(data_max_levels, electrodes=electrodes, n_zpos=resolution, axis='axial', verbose=False, device=device, resolution=resolution, return_aspect_ratio=True, z_padding=z_padding)
    pred_max = pred_max.squeeze()
    pred_min = reconstruct(data_min_levels, electrodes=electrodes, n_zpos=resolution, axis='axial', verbose=False, device=device, resolution=resolution, z_padding=z_padding).squeeze()
    preds = np.stack((pred_max, pred_min), axis=0)
    # plot
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))
    cbar_ax = fig.add_axes([0.93, 0.17, 0.03, 0.68])  # [left, bottom, width, height]
    plt.subplots_adjust(hspace=0.0)  

    axes[0,0].set_title(f'Inspiration')
    axes[0,1].set_title(f'Expiration')
    mask_scale = int(resolution/128)
    tomograms = []

    for j in range(2):
        tmp = preds[j,slice_axial]
        axes[0,j].imshow(tmp, cmap=cmap, vmin=0, vmax=0.7)
        tomograms.append(tmp)
        mask = np.zeros_like(preds[j,slice_axial])
        mask[50*mask_scale:90*mask_scale,83*mask_scale:98*mask_scale] = 1
        mask[50*mask_scale:90*mask_scale,27*mask_scale:42*mask_scale] = 1
        # axes[0,j].imshow(mask, alpha=0.1, cmap='Greys')
        axes[0,j].set_aspect(aspect_ratio[0])
        # axes[0,j].text(5, 5, f'Mean conductivity: {np.mean(preds[j,slice_axial][mask==1]):.3f}', fontsize=10)
        axes[0,j].axis('off')
        tmp = preds[j,:,slice_coronal]
        axes[1,j].imshow(tmp, cmap=cmap, vmin=0, vmax=0.7)
        tomograms.append(tmp)
        mask = np.zeros_like(preds[j,:,slice_coronal])
        mask[10*mask_scale:55*mask_scale,85*mask_scale:100*mask_scale] = 1
        mask[10*mask_scale:55*mask_scale,25*mask_scale:40*mask_scale] = 1
        # axes[1,j].imshow(mask, alpha=0.1, cmap='Greys')
        # axes[1,j].text(5, -1, f'Mean conductivity: {np.mean(preds[j,:,slice_coronal][mask==1]):.3f}', fontsize=10)
        axes[1,j].set_aspect(aspect_ratio[1])    
        axes[1,j].axis('off')

    # axes[0,0].set_ylabel('Axial')
    # axes[1,0].set_ylabel('Cornal')

    spacing = 0.4
    x_location = 0.08
    y_location = 0.7
    fig.text(
        x_location,
        y_location,
        f"Axial",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
        )
    fig.text(
        x_location,
        y_location - 1 * spacing,
        f"Coronal",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
    )
    #Add colorbar to the figure
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 0.7)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')        
    cbar.set_label('Conductivity (S/m)')
    plt.show()
    if save is not None:
        fig.savefig(save, bbox_inches="tight", dpi=600)
        plt.close(fig)
    else:
        return tomograms, aspect_ratio

def resist_single():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to data')    
    parser.add_argument('--output_dir', type=str, help='output directory', default='')
    parser.add_argument('--output_file', type=str, help='output filename', default='tomogram.png')
    parser.add_argument('--model', type=str, help='path to model', default='outputs/resist')
    parser.add_argument('--n_zpos', type=int, help='number of z positions', default=4)
    
    if parser.parse_args().output_dir=='':
        if os.path.isdir(parser.parse_args().data):
            output_path = parser.parse_args().data
        else:
            output_path = os.path.dirname(os.path.abspath(parser.parse_args().data))

    pred = reconstruct(data=parser.parse_args().data, model_path=parser.parse_args().model, n_zpos=parser.parse_args().n_zpos)

    fig, ax = plt.subplots(1, pred.shape[0], figsize=(6*pred.shape[0], 6))
    cbar_ax = fig.add_axes([0.93, 0.17, 0.03, 0.68])  # [left, bottom, width, height]
    plt.subplots_adjust(hspace=0.0)  

    for n in range(pred.shape[0]):
        ax[n].set_title(f'Level {str(n+1)}')
        ax[n].imshow(pred[n], cmap=cmap, vmin=0, vmax=0.7)
        ax[n].axis('off')
    #Add colorbar to the figure
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(0, 0.7)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')        
    cbar.set_label('Conductivity (S/m)')
    fig.savefig(os.path.join(output_path, parser.parse_args().output_file), bbox_inches='tight', dpi=600)