from vedo import *
import numpy as np
import os
import fnmatch

from data_processing.helper import rescale_img, get_mask_target_electrodes
from data_processing.obj2py import read_egt, read_get

def load_signals(return_square=True, dir=None):
    ''' 
    Loads the signals for the given case directory.
    '''    
    signal_dir  = os.path.join(dir,'signals')
    case_signal = []
    case_signals_files = os.listdir(signal_dir)
    case_signals_files = [f for f in case_signals_files if f.endswith('.get') & fnmatch.fnmatch(f, 'level_*_*.get')]
    case_signals_files.sort()
    
    for case_signal_file in case_signals_files:
        try:
            signal = read_get(signal_dir + '/' + case_signal_file)
            if return_square == True:
                signal_matrix = np.zeros((16,16))
                for i in range(16):
                    to_allocate = signal[i*13:(i+1)*13]
                    number_values_back = 16 - (i + 2)
                    number_values_back = 13 if number_values_back > 13 else number_values_back
                    number_values_back = 0 if number_values_back < 0 else number_values_back 
                    number_values_front = 13-number_values_back
                    signal_matrix[i, i+2:i+2+number_values_back] = to_allocate[:number_values_back]
                    signal_matrix[i, :number_values_front] = to_allocate[number_values_back:]
                signal = signal_matrix
            case_signal.append(signal)
        except:
            print(case_signal_file, 'can not be loaded.')
            
    case_signal = np.stack(case_signal,0)
    return case_signal

def load_tomograms(dir=None):
    ''' 
    Loads the tomograms for the given case directory.
    '''
    tomograms_dir = os.path.join(dir,'tomograms')     
    case_tomograms = []
    case_tomograms_files = os.listdir(tomograms_dir)
    case_tomograms_files = [f for f in case_tomograms_files if f.endswith('.egt')]
    case_tomograms_files.sort()
    for case_tomograms_file in case_tomograms_files:
        try:
            case_tomograms.append(read_egt(tomograms_dir + '/' + case_tomograms_file))
        except:
            print(case_tomograms_file, 'can not be loaded.')
    case_tomograms = np.stack(case_tomograms,0)
    return case_tomograms

# DB: Used for loading targets from .mat data - load_mask_target_electrodes() used now to load all from .nas/.vtk file aligned
# def load_targets(resolution=128, dir=None):
#     ''' 
#     Loads the targets for the given directory.
#     '''
#     targets_dir = os.path.join(dir,'targets')
#     case_targets = []
#     case_targets_files = os.listdir(targets_dir)
#     case_targets_files = [file for file in case_targets_files if file.endswith('.mat') & fnmatch.fnmatch(file, 'level_*_*.mat')]
#     case_targets_files.sort()
#     for case_targets_file in case_targets_files:
#         try:
#             mat = scipy.io.loadmat(targets_dir + '/' + case_targets_file)
#             key = list(mat.keys())[3]
#             target = mat[key]
#             target[np.isnan(target)] = 0
#             target = target.T
#             target = rescale_img(target)
#             target = cv2.resize(target, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
#             case_targets.append(target)
#         except:
#             print(case_targets_file, 'can not be loaded.')
#     case_targets = np.stack(case_targets,0)
#     return case_targets

def load_mask_target_electrodes(dir, resolution=128, electrode_resolution=512):
    ''' 
    Loads the masks for the given cases.
    '''
    dir = os.path.join(dir)
    mask, target, coord = get_mask_target_electrodes(dir, resolution=resolution, electrode_resolution=electrode_resolution)
    return mask, target, coord

def write_data(signals, targets, masks, electrodes, dir=None):
    np.savez_compressed(dir, signals=signals, targets=targets, masks=masks, electrodes=electrodes)

