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
                verbose=True):
    # get directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # load model and normalizing statistics
    if model_path is None:
        model_path = os.path.join(script_dir, 'outputs', 'resist')
        model_path = os.path.relpath(model_path,script_dir)
    model, cfg = load_model(model_path, device='cuda')
    signals_mean = torch.load(os.path.join(cfg.data.dataset_data_folder, 'train_dataset.pt')).train_mean.numpy()
    signals_std = torch.load(os.path.join(cfg.data.dataset_data_folder, 'train_dataset.pt')).train_std.numpy()
    points_max = torch.load(os.path.join(cfg.data.dataset_data_folder, 'train_dataset.pt')).points_max.numpy()
    points_min = torch.load(os.path.join(cfg.data.dataset_data_folder, 'train_dataset.pt')).points_min.numpy()

    # load electrodes position if available, otherwise use default
    if electrodes is None:
        if verbose:
            print('Using default electrode positions.')
        electrodes = np.load(os.path.join(parent_dir,'data/defaults/electrodes.npy'))
    else:
        electrodes = np.load(electrodes)
    
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

    electrodes = electrodes.reshape(-1, 16, 3)[:signals.shape[0]]
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
    elif axis == 'sagittal':
        resolution_x = n_zpos
    elif axis == 'coronal':
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
        z = np.linspace(max_z, min_z, resolution_z)
        xyz = np.meshgrid(x, y, z)
        # transpose grid to (z, y, x, 3) and switch x and y
        xyz = np.array(xyz).T
        xyz = np.moveaxis(xyz,1,1).reshape(1, -1, 3)
        points = np.tile(xyz, (batch_size, 1, 1))
    if batch_size > 1:
        pred_tmp = []
        for i in tqdm(range(batch_size)):
            _, pred, _ = testing(model, [signals[i][None], electrodes[i][None], points[i][None]], device=device, wandb_log=False, point_levels_3d=len(zpos), point_chunks=len(zpos))
            pred_tmp.append(pred.cpu())
        pred = torch.stack(pred_tmp, dim=0)
    else:
        _, pred, _ = testing(model, [signals, electrodes, points], device=device, wandb_log=False, point_levels_3d=len(zpos), point_chunks=len(zpos))
    pred = pred.reshape(batch_size, resolution_z, resolution_x, resolution_y).numpy()
    # move view axis to front
    if axis == 'axial':
        pred = np.moveaxis(pred, 1, 1)
    elif axis == 'sagittal':
        pred = np.moveaxis(pred, 2, 1)
    elif axis == 'coronal':
        pred = np.moveaxis(pred, 3, 1)

    return pred


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