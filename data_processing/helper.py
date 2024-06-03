from vedo import *
from vedo import Mesh
import scipy.io
import numpy as np
import os
import shapely.geometry as sg
import fnmatch
from tqdm import tqdm
import torch
import pyvista as pv
import pandas as pd
from data_processing.mesh_to_array import mesh_to_image, convert_to_vtk

def rescale_img(img, coords=None, resolution=512):
    '''
    Rescales the image to the given resolution by padding with zeros.
    If coords are given, the coordinates are also rescaled.
    '''
    y, x = np.where(img > 0)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    img = img[ymin:ymax, xmin:xmax]
    background = np.zeros((resolution, resolution))
    img_height, img_width = img.shape
    offset_height = int((resolution - img_height)/2)
    offset_width = int((resolution - img_width)/2)
    background[offset_height:offset_height+img_height, offset_width:offset_width+img_width] = img

    if coords is not None:
        coords[:,0] = coords[:,0] - xmin + offset_width
        coords[:,1] = coords[:,1] - ymin + offset_height
        return background, coords

    return background

def find_nearest(array, value):
    ''' 
    Returns the index of the element in array that is closest to value.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def sort_coordinates(list_of_xy_coords):
    '''
    Sorts the coordinates in a clockwise manner.
    '''
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]

def reduce_mask(mask):
    ''' 
    Reduces the mask to show the outer body shape only.
    '''
    y, x = np.where(mask > 0)
    coord = np.stack((x, y), axis=1)

    points_left = []
    points_right = []
    points_top = []
    points_bottom = []

    for xi in np.sort(np.unique(x)):
        coord_tmp = coord[coord[:, 0] == xi]
        ymin = np.argmin(coord_tmp[:, 1])
        ymax = np.argmax(coord_tmp[:, 1])
        points_bottom.append(coord_tmp[ymin, :])
        points_top.append(coord_tmp[ymax, :])
    points_top = np.stack(points_top, axis=0)
    points_bottom = np.stack(points_bottom, axis=0)

    for yi in np.sort(np.unique(y)):
        coord_tmp = coord[coord[:, 1] == yi]
        xmin = np.argmin(coord_tmp[:, 0])
        xmax = np.argmax(coord_tmp[:, 0])
        points_left.append(coord_tmp[xmin, :])
        points_right.append(coord_tmp[xmax, :])
    points_left = np.stack(points_left, axis=0)
    points_right = np.stack(points_right, axis=0)

    points = np.concatenate((points_top, np.flip(points_right, axis=0), np.flip(points_bottom, axis=0), points_left), axis=0)
    points = np.unique(points, axis=0)
    points_sorted = sort_coordinates(points)
    poly = sg.Polygon(points_sorted)
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if poly.contains(sg.Point(i+1, j+1)):
                mask[j, i] = 255
            else:
                mask[j, i] = 0
    return mask

def get_mask_target_electrodes(dir, resolution=128, electrode_resolution=512, mesh_from_nas=True, all_signals=False,
                                points_3d=False, point_levels_3d=8, point_range_3d=0.05):
    ''' 
    Returns the mask of the body shape and the coordinates of the electrodes.
    '''
    # TO DO: update function, remove unused arguments
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
        bounds = msh.bounds
        xlim = bounds[0:2]
        ylim = bounds[2:4]
        zlim = bounds[4:6]
    else:   
        file_path = os.path.join(dir,'shape/body_only.stl')
        if os.path.exists(file_path):
            msh = Mesh(file_path)
        else:
            file_path = os.path.join(dir,'shape/background.stl')
            msh = Mesh(file_path)
        zlim = msh.zbounds()
        xlim = msh.xbounds()
        ylim = msh.ybounds()

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

    # discretize the mesh
    slices = np.linspace(zlim[0], zlim[1], electrode_resolution)
    xrange = np.linspace(xlim[0], xlim[1], electrode_resolution)
    yrange = np.linspace(ylim[0], ylim[1], electrode_resolution)

    electrodes = electrodes.reshape(-1, 16, 3)
    # for e in electrodes:
    #     e[:,0] = np.array([find_nearest(xrange, ex) for ex in e[:,0]])
    #     e[:,1] = np.array([find_nearest(yrange, ey) for ey in e[:,1]])

    z_positions = [np.mean(e[:,2]) for e in electrodes]
    if all_signals:
        z_positions = create_equal_distant_array(n=point_levels_3d, value1=z_positions[0], value2=z_positions[1], value3=z_positions[2], value4=z_positions[3])
    mesh_arrays = [mesh_to_image(msh, z_pos=z, resolution=resolution) for z in z_positions] 
    points = [point[1] for point in mesh_arrays] # shape: (point_levels_3d * 512 * 512, 3)
    center_vector = mesh_arrays[0][2]
    # use mean z_position as center point along z-axis
    center_vector[2] = np.mean(z_positions)
    targets = [target[0] for target in mesh_arrays]
    targets = np.stack(targets, axis=0)
    points = np.stack(points, axis=0) - center_vector
    electrodes = electrodes - center_vector
    # bounds_mean = np.asarray(bounds).reshape(3,2).mean(axis=1).reshape(1, 1, 3)
    # electrodes = electrodes - bounds_mean
    # electrodes[:,:,:2] = electrodes[:,:,:2] / (electrode_resolution/resolution)
    return targets, electrodes, points

def change_rho(targets, lung_rhos):
    class_resistancies = np.array([0., 0.3, 0, 0.7, 0.02, 0.025])
    targets_rho = []
    for i,lung_rho in enumerate(lung_rhos):
        class_resistancies[2] = 1/float(lung_rho)
        targets_rho.append(class_resistancies[targets])
    targets_rho = np.stack(targets_rho, axis=0)
    return targets_rho

def flatten_data(signals, targets, masks, electrodes, as_tensor=True):
    ''' 
    Flattens the data from the case to the image level.
    '''
    signals = np.concatenate(signals,0)
    targets = np.concatenate(targets,0)
    masks = np.concatenate(masks,0)
    electrodes = np.concatenate(electrodes,0)
    if as_tensor:
        signals = torch.from_numpy(signals).float()
        targets = torch.from_numpy(targets).float()
        masks = torch.from_numpy(masks).float()
        electrodes = torch.from_numpy(electrodes).float()
    return signals, targets, masks, electrodes

def cosine_similarity(impulses, points):
    points_unit = torch.nn.functional.normalize(points, p=2, dim=1)
    impulses_unit = torch.nn.functional.normalize(impulses, p=2, dim=1)
    euclidean_dist = torch.cdist(impulses_unit.unsqueeze(0), points_unit.unsqueeze(0), p=2).squeeze()
    euclidean_dist = 1 - (euclidean_dist**2) / 2
    return euclidean_dist.T

def combine_electrode_positions(electrodes, use_epair_center=False):
    # concatenate each row with the next row
    if use_epair_center:
        return_dim = 2
    else:
        return_dim = 4
    new_electrodes = np.zeros((electrodes.shape[0], 16, 13, return_dim, 3))
    for i in range(16):
        for j in range(13):
            if use_epair_center:
                new_electrodes[:,i,j,0] = electrodes[:,i]
                new_electrodes[:,i,j,1] = electrodes[:,(i+2+j)%16]
            else:                
                new_electrodes[:,i,j,0] = electrodes[:,i]
                new_electrodes[:,i,j,1] = electrodes[:,(i+1)%16]
                new_electrodes[:,i,j,2] = electrodes[:,(i+2+j)%16]
                new_electrodes[:,i,j,3] = electrodes[:,(i+3+j)%16]
    return new_electrodes


def create_equal_distant_array(n, value1, value2, value3, value4):
    '''
    Create an equal distant array between value1 and value2 and value3 and value4 and adds an additional point at the beginning and end.
    '''
    if n < 6:
        raise ValueError("n must be at least 5 to place values correctly")
    if n % 3 != 0:
        raise ValueError("n must be divisible by 3 to place values correctly")

    if value1 <= value2 or value2 <= value3 or value1 <= value3:
        raise ValueError("Check the z-coordinates, as the values are not decreasing in their values")

    # Make an equal distant array between value1 and value2 and value2 and value3
    n_create = n // 3 # we would have to reduce the number by 2 and then distribute the number over 3 sub arrays (v1-v2, v2-v3, v3-v4), we use v2, v3 as start point so no reduction necessary
    array1 = np.linspace(value1, value2, int(n_create))
    array2 = np.linspace(value2, value3, int(n_create))
    array3 = np.linspace(value3, value4, int(n_create))
    dist_begin = np.abs(array1[1] - array1[0])
    dist_end = np.abs(array3[1] - array3[0])
    array = np.concatenate((np.array([value1 + dist_begin]), array1, array2[1:], array3[1:], np.array([value4 - dist_end])))
    return array