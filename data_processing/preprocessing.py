from vedo import *
import numpy as np
import os
import fnmatch
import pandas as pd
import pyvista as pv
import scipy
from data_processing.helper import create_equal_distant_array, pad_to_shape_centered, extract_float_rho
from data_processing.mesh_to_array import mesh_to_image, convert_to_vtk, mesh_to_voxels
from data_processing.obj2py import read_egt, read_get

def load_signals(return_square=True, dir=None):
    ''' 
    Loads the signals for the given case directory.
    '''    
    signal_dir  = os.path.join(dir,'signals')
    case_signal = []
    rhos = []
    level = []
    case_signals_files = os.listdir(signal_dir)
    case_signals_files = [f for f in case_signals_files if fnmatch.fnmatch(f, 'level_*_*.*')]
    def custom_sort(s):
        idx1 = int(s.split('_')[1]) 
        idx2 = int(s.split('_')[2].split('.')[0]) 
        return idx1, idx2
    case_signals_files = sorted(case_signals_files, key=custom_sort)
    
    for case_signal_file in case_signals_files:
        try:
            if case_signal_file.count('_') == 3:
                if case_signal_file.endswith('.get'):
                    signal = read_get(signal_dir + '/' + case_signal_file)
                elif case_signal_file.endswith('.npy'):
                    signal = np.load(signal_dir + '/' + case_signal_file)
                else:
                    print(f'Check signal files for {case_signal_file}!')
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
                else:
                    signal = signal.reshape(16,13)
                # rhos.append(int(case_signal_file.split('_')[2].split('.')[0]))
                rhos.append(extract_float_rho(case_signal_file))
                level.append(int(case_signal_file.split('_')[1]))
                case_signal.append(signal)
        except:
            print(case_signal_file, 'can not be loaded.')
            
    case_signal = np.stack(case_signal,0)
    rhos = np.array(rhos)
    level = np.array(level)
    return case_signal, rhos, level

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

def load_targets_electrodes_points(dir, resolution=128, mesh_from_nas=True, point_levels_3d=8):
    ''' 
    Returns the targets (categories), corresponding points andcoordinates of the electrodes.
    '''
    # load raw stl file 
    if mesh_from_nas:
        file_path = os.path.join(dir, 'shape')
        shapes = os.listdir(file_path)
        shapes = [shape for shape in shapes if fnmatch.fnmatch(shape, '*.vtk')]
        if len(shapes)==0:
            print(f'No .vtk file found in {dir}. Search .nas file', end='...')            
            shapes = os.listdir(file_path)
            shapes = [shape for shape in shapes if fnmatch.fnmatch(shape, '*.nas')]
            if len(shapes)==0:
                raise ValueError(f'No shape file found in {file_path}')
            file_path = os.path.join(file_path, shapes[0])
            convert_to_vtk(file_path)
            file_path = file_path.split('.nas')[0]+'.vtk'
        else:
            file_path = os.path.join(file_path, shapes[0])
        msh = pv.read(file_path)
    else:   
        file_path = os.path.join(dir,'shape/body_only.stl')
        if os.path.exists(file_path):
            msh = Mesh(file_path)
        else:
            file_path = os.path.join(dir,'shape/background.stl')
            msh = Mesh(file_path)

    # load raw electrode coordinates   
    electrodes_from_mat = os.path.exists(os.path.join(dir,'electrodes/electrodes.mat'))

    if electrodes_from_mat:
        electrodes_path = os.path.join(dir,'electrodes/electrodes.mat')
        mat = scipy.io.loadmat(electrodes_path)
        key = list(mat.keys())[3]
        electrodes = mat[key]
        # every second point was used
        electrodes = electrodes[::2]
    else:
        electrodes_path = os.path.join(dir,'electrodes/electrodes.txt')
        if not os.path.exists(electrodes_path):
            folder_path = os.path.split(electrodes_path)[0]
            electrodes_path = os.listdir(folder_path)[0]
            print(f'Electrode file not found. Try to load {electrodes_path} instead.')
            electrodes_path = os.path.join(folder_path, electrodes_path)
        electrodes = pd.read_csv(electrodes_path, sep=" ", header=None).values[:-1]

    electrodes = electrodes.reshape(-1, 16, 3)

    z_positions = [np.mean(e[:,2]) for e in electrodes]
    z_positions = create_equal_distant_array(n=point_levels_3d, value1=z_positions[0], value2=z_positions[1], value3=z_positions[2], value4=z_positions[3])
    mesh_arrays = [mesh_to_image(msh, z_pos=z, resolution=resolution) for z in z_positions] 
    points = [point[1] for point in mesh_arrays] # shape: (point_levels_3d * 512 * 512, 3)
    center_vector = mesh_arrays[0][2]
    # use mean z_position as center point along z-axis
    center_vector[2] = np.mean(z_positions)
    center_vector = electrodes[0,0]
    targets = [target[0] for target in mesh_arrays]
    targets = np.stack(targets, axis=0)
    points = np.stack(points, axis=0) - center_vector
    electrodes = electrodes - center_vector

    return targets, electrodes, points

def load_body_shape(dir, density=0.01):
    # get paths for mesh and electrodes
    file_path = os.path.join(dir, 'shape')
    shapes = os.listdir(file_path)
    shapes = [shape for shape in shapes if fnmatch.fnmatch(shape, '*.vtk')]
    mesh_path = os.path.join(file_path, shapes[0])
    electrodes_path = os.path.join(dir,'electrodes/electrodes.txt')
    coords = pd.read_csv(electrodes_path, sep=" ", header=None).values[:-1]
    coords = coords.reshape(-1, 16, 3)
    # get the z coordinates of the levels
    z_coords = np.mean(coords, axis=1)[:,2]
    # load mesh and clip it to be close to the electrode levels
    mesh_in = pv.read(mesh_path)
    xmin, xmax = mesh_in.bounds[:2]
    ymin, ymax = mesh_in.bounds[2:4]
    zmin, zmax = mesh_in.bounds[4:6]
    z_end, z_start = create_equal_distant_array(z_coords.shape[0]+2, z_coords[0], z_coords[1], z_coords[2], z_coords[3])[[0,-1]]
    mesh_in = mesh_in.clip_box([xmin, xmax, ymin, ymax, z_start, z_end], invert=False)
    xmin, xmax = mesh_in.bounds[:2]
    ymin, ymax = mesh_in.bounds[2:4]
    zmin, zmax = mesh_in.bounds[4:6]
    mesh_in = mesh_in.clip_box([xmin, xmax, ymin, ymax, z_start, z_end], invert=False)
    # get the array from the mesh to obtain the voxel masks
    mask = mesh_to_voxels(mesh_in, density)
    # pad mask to the same shape for all cases
    mask = pad_to_shape_centered(mask, target_shape=(256,256,256))
    return mask