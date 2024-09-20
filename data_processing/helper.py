from vedo import *
from vedo import Mesh
import scipy.io
import numpy as np
import os
import shapely.geometry as sg
import fnmatch
from tqdm import tqdm
import re
import torch
import pyvista as pv
import pandas as pd
from data_processing.mesh_to_array import mesh_to_image, convert_to_vtk
from scipy.ndimage import binary_erosion

def extract_float_rho(string):
    '''
    Used to get the rho value from the filename 'level_<level>_rho_<rho>.get'
    '''
    # Split by '_' and '.' to extract the required parts
    parts = string.split('_')
    
    # Convert value1 to an integer and value2 to a float
    value1 = int(parts[2])
    value2 = int(parts[3].split('.')[0])
    
    # Combine them into a single float where value1 is before the decimal and value2 is after
    combined_value = float(f"{value1}.{value2}")
    
    return combined_value

def rescale_img(img, coords=None, resolution=512):
    """
    Rescales the image to the given resolution by padding with zeros.
    If coords are given, the coordinates are also rescaled.
    """
    y, x = np.where(img > 0)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    img = img[ymin:ymax, xmin:xmax]
    background = np.zeros((resolution, resolution))
    img_height, img_width = img.shape
    offset_height = int((resolution - img_height) / 2)
    offset_width = int((resolution - img_width) / 2)
    background[
        offset_height : offset_height + img_height,
        offset_width : offset_width + img_width,
    ] = img

    if coords is not None:
        coords[:, 0] = coords[:, 0] - xmin + offset_width
        coords[:, 1] = coords[:, 1] - ymin + offset_height
        return background, coords

    return background


def find_nearest(array, value):
    """
    Returns the index of the element in array that is closest to value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def sort_coordinates(list_of_xy_coords):
    """
    Sorts the coordinates in a clockwise manner.
    """
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]


def reduce_mask(mask):
    """
    Reduces the mask to show the outer body shape only.
    """
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

    points = np.concatenate(
        (
            points_top,
            np.flip(points_right, axis=0),
            np.flip(points_bottom, axis=0),
            points_left,
        ),
        axis=0,
    )
    points = np.unique(points, axis=0)
    points_sorted = sort_coordinates(points)
    poly = sg.Polygon(points_sorted)
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if poly.contains(sg.Point(i + 1, j + 1)):
                mask[j, i] = 255
            else:
                mask[j, i] = 0
    return mask


def change_rho(targets, lung_rhos):
    class_resistancies = np.array([0.0, 0.3, 0, 0.7, 0.02, 0.025])
    targets_rho = []
    for i, lung_rho in enumerate(lung_rhos):
        class_resistancies[2] = 1 / float(lung_rho)
        targets_rho.append(class_resistancies[targets])
    targets_rho = np.stack(targets_rho, axis=0)
    return targets_rho


def change_rho_single(targets, lung_rho):
    class_resistancies = np.array([0.0, 0.3, 0, 0.7, 0.02, 0.025])
    class_resistancies[2] = 1 / float(lung_rho)
    targets_rho = np.stack(targets_rho, axis=0)
    return targets_rho


def flatten_data(signals, targets, masks, electrodes, as_tensor=True):
    """
    Flattens the data from the case to the image level.
    """
    signals = np.concatenate(signals, 0)
    targets = np.concatenate(targets, 0)
    masks = np.concatenate(masks, 0)
    electrodes = np.concatenate(electrodes, 0)
    if as_tensor:
        signals = torch.from_numpy(signals).float()
        targets = torch.from_numpy(targets).float()
        masks = torch.from_numpy(masks).float()
        electrodes = torch.from_numpy(electrodes).float()
    return signals, targets, masks, electrodes


def cosine_similarity(impulses, points):
    points_unit = torch.nn.functional.normalize(points, p=2, dim=1)
    impulses_unit = torch.nn.functional.normalize(impulses, p=2, dim=1)
    euclidean_dist = torch.cdist(
        impulses_unit.unsqueeze(0), points_unit.unsqueeze(0), p=2
    ).squeeze()
    euclidean_dist = 1 - (euclidean_dist**2) / 2
    return euclidean_dist.T


def combine_electrode_positions(electrodes, use_epair_center=False):
    '''
    Stacks recording and feeding electrodes
    '''
    # concatenate each row with the next row
    if use_epair_center:
        return_dim = 2
    else:
        return_dim = 4
    new_electrodes = np.zeros((electrodes.shape[0], 16, 13, return_dim, 3))
    for i in range(16):
        for j in range(13):
            if use_epair_center:
                new_electrodes[:, i, j, 0] = electrodes[:, i]
                new_electrodes[:, i, j, 1] = electrodes[:, (i + 2 + j) % 16]
            else:
                new_electrodes[:, i, j, 0] = electrodes[:, i]
                new_electrodes[:, i, j, 1] = electrodes[:, (i + 1) % 16]
                new_electrodes[:, i, j, 2] = electrodes[:, (i + 2 + j) % 16]
                new_electrodes[:, i, j, 3] = electrodes[:, (i + 3 + j) % 16]
    return new_electrodes


def create_equal_distant_array(n, value1, value2, value3, value4):
    """
    Create an equal distant array between value1 and value2 and value3 and value4 and adds an additional point at the beginning and end.
    """
    if n == 4:
        return np.array([value1, value2, value3, value4])
    elif n < 6:
        raise ValueError("n must be at least 6 to place values correctly")
    if n % 3 != 0:
        raise ValueError(
            "n must be divisible by 3 and greater than 3 to place values correctly"
        )
    if value1 <= value2 or value2 <= value3 or value1 <= value3:
        raise ValueError(
            "Check the z-coordinates, as the values are not decreasing in their values"
        )

    # Make an equal distant array between value1 and value2 and value2 and value3
    n_create = (
        n // 3
    )  # we would have to reduce the number by 2 and then distribute the number over 3 sub arrays (v1-v2, v2-v3, v3-v4), we use v2, v3 as start point so no reduction necessary
    array1 = np.linspace(value1, value2, int(n_create))
    array2 = np.linspace(value2, value3, int(n_create))
    array3 = np.linspace(value3, value4, int(n_create))
    dist_begin = np.abs(array1[1] - array1[0])
    dist_end = np.abs(array3[1] - array3[0])
    array = np.concatenate(
        (
            np.array([value1 + dist_begin]),
            array1,
            array2[1:],
            array3[1:],
            np.array([value4 - dist_end]),
        )
    )
    return array


def pad_to_shape_centered(array, target_shape, padding_value=0):
    """
    Pad a 3D numpy array to fit a specified resolution in 3D, centering the original array.

    Parameters:
    array (np.ndarray): The input 3D array to pad.
    target_shape (tuple): The desired shape of the output array (z, y, x).
    padding_value (int, float, optional): The value to use for padding. Defaults to 0.

    Returns:
    np.ndarray: The padded array.
    """
    # Current shape of the input array
    current_shape = array.shape

    # Calculate padding for each dimension to center the original array
    pad_z1 = (target_shape[0] - current_shape[0]) // 2
    pad_z2 = target_shape[0] - current_shape[0] - pad_z1

    pad_y1 = (target_shape[1] - current_shape[1]) // 2
    pad_y2 = target_shape[1] - current_shape[1] - pad_y1

    pad_x1 = (target_shape[2] - current_shape[2]) // 2
    pad_x2 = target_shape[2] - current_shape[2] - pad_x1

    # Apply padding
    padded_array = np.pad(
        array,
        ((pad_z1, pad_z2), (pad_y1, pad_y2), (pad_x1, pad_x2)),
        mode="constant",
        constant_values=padding_value,
    )
    return padded_array


# sort list of strings by case first and then by resistivity
def extract_keys(filename):
    """
    Extract the third-to-last and last numerical keys from a filename of the format 'abc_x_y_key.npz'.

    Parameters:
    filename (str): The filename to extract the keys from.

    Returns:
    tuple: A tuple containing the third-to-last and last numerical keys.
    """
    # Split the filename by underscores
    parts = filename.split("_")
    # Extract the third-to-last part (key1) and the last part (key2 with .npz extension)
    key1 = int(parts[-3])
    key2_with_extension = parts[-1]
    # Remove the .npz extension to get the key2
    key2 = int(key2_with_extension.split(".")[0])
    return key1, key2


def sort_filenames(filenames):
    """
    Sort a list of filenames by the third-to-last key first, and then by the last key.

    Parameters:
    filenames (list of str): The list of filenames to sort.

    Returns:
    list of str: The sorted list of filenames.
    """
    return sorted(filenames, key=extract_keys)


def erode_lung_masks(targets, conductivity=None, structure=30):
    '''
    Extracts the lung mask from target for a specific conductivity and erodes it with a binary erosion
    '''
    if conductivity is None:
        lung_masks = (targets <= 0.2) * (targets >= 0.05)
    else:
        lung_masks = targets == conductivity
    lung_masks = lung_masks.squeeze()
    if isinstance(lung_masks, torch.Tensor):
        lung_masks = lung_masks.numpy()
    lung_masks = [
        binary_erosion(lung_mask, structure=np.ones((structure, structure))).astype(
            np.uint8
        )
        for lung_mask in lung_masks
    ]
    lung_masks = torch.tensor(np.array(lung_masks))
    return lung_masks

def extract_rho_number(strings):
    '''
    Extracts the rho number from a list of strings
    '''
    pattern = re.compile(r"rho_(\d*\.\d+|\d+)")
    extracted = [
        float(pattern.search(string).group(1)) if pattern.search(string) else None
        for string in strings
    ]
    return extracted


def extract_level_number(strings, processed=False):
    '''
    Extracts the level number from a list of strings
    '''
    if processed:
        pattern = re.compile(r"level_(\d+)_")
    else:
        pattern = re.compile(r"rho_(?:\d*\.\d+|\d+)+_z(\d+)")
    extracted = [
        int(pattern.search(string).group(1)) if pattern.search(string) else None
        for string in strings
    ]
    return extracted


def sort_strings_by_numbers(strings, first_list, second_list):
    ''''
    Sorts a list of strings and two values by the the first and then by the second value
    '''

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
    """
    Takes a "signals" directory and returns the sorted path to each measurement,
    the body level and the used lung rho as lists
    """
    files = os.listdir(dir)
    files = [signal for signal in files if signal.endswith(".get")]
    rhos = extract_rho_number(files)
    files = [files[i] for i in range(len(files)) if rhos[i] is not None]
    rhos = [r for r in rhos if r is not None]
    levels = extract_level_number(files)
    files = [files[i] for i in range(len(files)) if levels[i] is not None]
    levels = [l for l in levels if levels is not None]
    print(files, levels, rhos)
    files, levels, rhos = sort_strings_by_numbers(files, levels, rhos)
    files = [os.path.join(dir, f) for f in files]
    return files, levels, rhos


def move_to_level_summary(dir):
    """
    Takes a "signals" directory and moves all files which have the pattern *_<rho1>_<rho2>_<level>
    to a level summary directory and renames them accordingly
    """
    strings = os.listdir(dir)
    level_summary_path = os.path.join(dir, "level_summary")
    os.makedirs(level_summary_path, exist_ok=True)
    pattern_1 = re.compile(r"_(\d+)_(\d+)_z(\d+)")
    pattern_2 = re.compile(r"_Liste_z(\d+)")

    for string in strings:
        match = pattern_1.search(string)
        if match:
            number1 = match.group(1)
            number2 = match.group(2)
            number3 = match.group(3)
            dest_file_name = f"level_{number3}_rho_{number1}_to_{number2}.get"
            dest_file = os.path.join(level_summary_path, dest_file_name)
            source_file = os.path.join(dir, string)
            os.rename(source_file, dest_file)
        else:
            match = pattern_2.search(string)
            if match:
                number = match.group(1)
                dest_file_name = f"level_{number}.get"
                dest_file = os.path.join(level_summary_path, dest_file_name)
                source_file = os.path.join(dir, string)
                os.rename(source_file, dest_file)