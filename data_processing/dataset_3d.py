import torch
from torch.utils.data import Dataset
from data_processing.helper import cosine_similarity, change_rho, combine_electrode_positions
import os
import fnmatch
from tqdm import tqdm
import cv2
import numpy as np
from data_processing.preprocessing import load_signals, write_data
from data_processing.helper import get_mask_target_electrodes
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
from utils.helper import set_seeds
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools


def make_npz_3d_multi(cases, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=512, electrode_resolution=512, overwrite_npz=False,
                points_3d=False, point_levels_3d=9, point_range_3d=0.05, num_workers='all', all_signals=False):
    # use all cpu cores if not specified
    num_workers = os.cpu_count() if num_workers=='all' else num_workers
    print(f'Using {num_workers} workers.')
    # exclude existing cases
    cases_copy = cases.copy()
    for case in cases:
        if os.path.exists(os.path.join(processed_data_folder, case)) and not overwrite_npz:
            cases_copy.remove(case)
    if len(cases_copy) == 0:
        return 'No cases to write to .npz.'
    partial_function = functools.partial(write_npz_case_3d, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                                         resolution=resolution, electrode_resolution=electrode_resolution, points_3d=points_3d, point_levels_3d=point_levels_3d, 
                                         point_range_3d=point_range_3d, multi=True, all_signals=all_signals)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(partial_function, item) for item in cases_copy]
        # Collect results with a progress bar
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            results.append(result)

def write_npz_case_3d(case, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=128, electrode_resolution=512,
                      mask_resolution=512, points_3d=False, point_levels_3d=9, point_range_3d=0.05, multi=False, all_signals=False):
    # how many levels and how many measurements per level do we have?
    if isinstance(case, int):
        case = 'case_'+str(case)
    if point_levels_3d % 3 != 0:
        print(f'point_levels_3d must be odd to include the level where the electrodes were placed. Increase point_levels_3d by one to {point_levels_3d+(3-(point_levels_3d%3))}.')
        point_levels_3d = point_levels_3d+1
    case_dir = os.path.join(raw_data_folder, case)
    case_dir_processed = os.path.join(processed_data_folder, case)
    
    signal_dir = os.path.join(case_dir, 'signals')
    case_signals_files = os.listdir(signal_dir)
    case_signals_files = [file for file in case_signals_files if fnmatch.fnmatch(file, 'level_*_*.*')]
    def custom_sort(s):
        idx1 = int(s.split('_')[1]) 
        idx2 = int(s.split('_')[2].split('.')[0]) 
        return idx1, idx2
    case_signals_files = sorted(case_signals_files, key=custom_sort)
    # get the level number
    case_signals_files_short = [int(f[6]) for f in case_signals_files]
    # get the lung rho
    case_signals_rhos = [f.split('_')[-1].split('.')[0] for f in case_signals_files]
    # get the level and the number of simulations per level
    levels, level_counts = np.unique(case_signals_files_short, return_counts=True)
    n_levels = levels.shape[0]
    n_rhos = np.unique(case_signals_rhos).shape[0]
    # load data
    signal = load_signals(dir=case_dir, return_square=False)  
    signal = np.moveaxis(signal.reshape(n_levels, n_rhos, -1), 1, 0)
    case_signals_rhos = np.moveaxis(np.array(case_signals_rhos).reshape(n_levels, n_rhos), 1, 0)
    targets, electrode, points = get_mask_target_electrodes(case_dir, resolution=resolution, electrode_resolution=electrode_resolution, all_signals=all_signals,
                                                                   points_3d=points_3d, point_levels_3d=point_levels_3d, point_range_3d=point_range_3d)

    # define desired shapes for all tensors
    target_shape = (n_rhos, point_levels_3d, resolution, resolution) 
    point_shape = (point_levels_3d, -1, 3)
    electrode_shape = (n_levels, -1, 3)
    signals_shape = (n_rhos, n_levels, -1)    

    case_signals_rhos = case_signals_rhos[:,0]
    # targets = targets.repeat(n_rhos, axis=0).reshape(case_signals_rhos.shape[0], -1, resolution, resolution)
    targets = change_rho(targets, case_signals_rhos)

    points = points.reshape(point_shape)
    targets = targets.reshape(target_shape)
    signal = signal.reshape(signals_shape)

    # assign electrodes the right level (up - down)
    level = np.mean(electrode[:,:,-1], axis=1)
    level = np.argsort(level)
    # max_level, min_level = np.max(electrode[:,:,-1]), np.min(electrode[:,:,-1])
    # electrode[:,:,-1] = ((electrode[:,:,-1] - min_level) / (max_level - min_level)) * 2 - 1
    electrode = electrode.reshape(electrode_shape)
    electrode = combine_electrode_positions(electrode) 
    # resize mask
    mask = np.where(targets>0, 1, 0)

    # make directory for each case
    os.makedirs(os.path.join(case_dir_processed), exist_ok=True)
    for s, t, r in zip(signal, targets, case_signals_rhos):
        np.savez_compressed(os.path.join(case_dir_processed,case+'_'+r+'.npz'), signals=s, targets=t, masks=mask[0], electrodes=electrode, points=points)

def make_npz_3d(cases, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=512, electrode_resolution=512, overwrite_npz=False,
                points_3d=False, point_levels_3d=8, point_range_3d=0.05, all_signals=False):
    with tqdm(cases) as pbar:
        for case in pbar:
            pbar.set_description(f"Processing {case}.")
            if os.path.exists(os.path.join(processed_data_folder, case+'_3d.npz')) and not overwrite_npz:
                continue
            # try:
            write_npz_case_3d(case, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                            resolution=resolution, electrode_resolution=electrode_resolution, all_signals=all_signals,
                            points_3d=points_3d, point_levels_3d=point_levels_3d, point_range_3d=point_range_3d)
            # except Exception as e:
            #     print(case, 'can not be processed to .npz file.')
            #     print(e)

def make_dataset_3d(cases, processed_data_folder="data/processed", base_dir='',resolution=512, mask_resolution=512, 
                n_sample_points=10000, return_electrodes=True, points_3d=True, point_levels_3d=9,
                 path_test_dataset="data/datasets/test_dataset_3d.pt", path_val_dataset="data/datasets/val_datasett_3d.pt", 
                 path_train_dataset="data/datasets/train_datasett_3d.pt"):
    set_seeds(123)

    # split into training, validation and test
    # use all cases up to 100 as test set
    cases_number = [int(case.split('_')[-2]) for case in cases]
    cases_number = [case_number for case_number in cases_number if case_number>400 and case_number<450]
    cases_number.sort()
    test_cases = ['case_TCIA_'+str(case_number)+'_0' for case_number in cases_number]
    # remove test cases
    cases = [case for case in cases if case not in test_cases]
    number_cases = len(cases)  
    number_training_cases = int(number_cases*0.9)
    number_validation_cases = int(number_cases*0.1)
    number_test_cases = len(test_cases) 

    # only save case number for data set
    train_cases = cases[:number_training_cases]
    val_cases = cases[number_training_cases:]

    # get some characteristics by loading certain parts of the data
    # - find the max coordinates loading all .npz files
    # - find the mean and std of the signals (training cases only)
    signals = []
    for i, case in tqdm(enumerate(cases)):
        file = os.listdir(os.path.join(processed_data_folder, case))[0]
        file_path = os.path.join(processed_data_folder, case, file)
        points = np.load(file_path)['points']
        if i==0:
            # old: points[:,:,:2].max()
            max_coord = points.max()
            min_coord = points.min() 
        else:
            if max_coord < points.max():
                max_coord = points.max()
            if min_coord > points.min():
                min_coord = points.min()
        if case in train_cases:
            signals.append(np.load(file_path)['signals'])
    # normalize signals
    train_signals = torch.from_numpy(np.concatenate(signals, axis=0))
    train_signal_mean = train_signals.mean(dim=(0,1), keepdim=True)
    train_signal_std = train_signals.std(dim=(0,1), keepdim=True)

    train_dataset = EITData3D(train_cases, resolution=resolution, training=True, n_sample_points=n_sample_points, train_mean=train_signal_mean, train_std=train_signal_std, 
                              points_3d=points_3d, point_levels_3d=point_levels_3d, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord, 
                              processed_path=processed_data_folder, base_dir=base_dir)
    val_dataset = EITData3D(val_cases, resolution=resolution, training=False, points_3d=points_3d, point_levels_3d=point_levels_3d,
                            train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord,
                            processed_path=processed_data_folder, base_dir=base_dir)
    test_dataset = EITData3D(test_cases, resolution=resolution, training=False, points_3d=points_3d, point_levels_3d=point_levels_3d,                             
                            train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord,
                            processed_path=processed_data_folder, base_dir=base_dir)
    print(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}')

    torch.save(train_dataset, path_train_dataset)
    torch.save(val_dataset, path_val_dataset)
    torch.save(test_dataset, path_test_dataset)

    # # normalize signals
    # train_signal_mean = train_signals.mean(dim=(0,1), keepdim=True)
    # train_signal_std = train_signals.std(dim=(0,1), keepdim=True)
    # # train_signal_mean = train_signals.mean()
    # # train_signal_std = train_signals.std()
    # train_signals = (train_signals - train_signal_mean) / train_signal_std
    # test_signals = (test_signals - train_signal_mean) / train_signal_std
    # val_signals = (val_signals - train_signal_mean) / train_signal_std
    # # normalize points and electrode position
    # train_points = ((train_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    # val_points = ((val_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    # test_points = ((test_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    # train_electrodes[:,:,:2] = ((train_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1
    # val_electrodes[:,:,:2] = ((val_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1
    # test_electrodes[:,:,:2] = ((test_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1


    # train_dataset = EITData3D(train_signals, train_targets, train_masks, train_electrodes, train_levels, train_cases, train_points, resolution=resolution, training=True, 
    #                         n_sample_points=n_sample_points, train_mean=train_signal_mean, train_std=train_signal_std, points_3d=points_3d, point_levels_3d=point_levels_3d,
    #                         return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord)
    # val_dataset = EITData3D(val_signals, val_targets, val_masks, val_electrodes, val_levels, val_cases, val_points, resolution=resolution, training=False,
    #                         points_3d=points_3d, point_levels_3d=point_levels_3d,
    #                         train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord)
    # test_dataset = EITData3D(test_signals, test_targets, test_masks, test_electrodes, test_levels, test_cases, test_points, resolution=resolution, training=False,
    #                         points_3d=points_3d, point_levels_3d=point_levels_3d,                             
    #                         train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord)
    # print(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}')

    # torch.save(train_dataset, path_train_dataset)
    # torch.save(val_dataset, path_val_dataset)
    # torch.save(test_dataset, path_test_dataset)


def load_dataset_3d(cases, resolution=512, electrode_resolution=512, mask_resolution=512, n_sample_points=10000,
                 base_dir="", raw_data_folder="data/raw", processed_data_folder="data/processed", dataset_data_folder="data/datasets", 
                 no_weights=False, name_prefix="", write_dataset=False, write_npz=False, overwrite_npz=False, return_electrodes=True,
                 apply_rotation=False, apply_subsampling=True, use_epair_center=False, points_3d=False, point_levels_3d=9, point_range_3d=0.05,
                 num_workers=None, multi_process=True, all_signals=False):
    '''
    Returns:
        train_dataset, val_dataset, test_dataset with:
            signals: torch.Tensor of shape (n_cases*n_levels*n_rhos, 16, 13, )
            targets: torch.Tensor of shape (n_cases*n_levels*n_rhos, point_level_3d, resolution, resolution)
            masks: torch.Tensor of shape (n_cases*n_levels*n_rhos, point_level_3d, resolution, resolution)
            electrodes: torch.Tensor of shape (n_cases*n_levels*n_rhos, n_electrodes, 3)
            levels: torch.Tensor of shape (n_cases*n_levels*n_rhos)
            points: torch.Tensor of shape (n_cases*n_levels*n_rhos, point_level_3d, resolution*resolution, 3)
            cases: list of str
    '''
    # set up paths
    raw_data_folder = os.path.normpath(os.path.join(base_dir,raw_data_folder))
    processed_data_folder = os.path.normpath(os.path.join(base_dir,processed_data_folder))
    dataset_data_folder = os.path.normpath(os.path.join(base_dir,dataset_data_folder))        
    path_train_dataset = os.path.join(dataset_data_folder,'train_dataset_3d'+name_prefix+'.pt')
    path_val_dataset = os.path.join(dataset_data_folder,'val_dataset_3d'+name_prefix+'.pt')
    path_test_dataset = os.path.join(dataset_data_folder,'test_dataset_3d'+name_prefix+'.pt')

    if write_npz:
        if multi_process:
            make_npz_3d_multi(cases, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                 resolution=resolution, electrode_resolution=electrode_resolution, overwrite_npz=overwrite_npz,
                 points_3d=points_3d, point_levels_3d=point_levels_3d, point_range_3d=point_range_3d, all_signals=all_signals,
                 num_workers=num_workers)
        else:
            make_npz_3d(cases, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                 resolution=resolution, electrode_resolution=electrode_resolution, overwrite_npz=overwrite_npz,
                 points_3d=points_3d, point_levels_3d=point_levels_3d, point_range_3d=point_range_3d, all_signals=all_signals)
    if write_dataset:
        make_dataset_3d(cases, processed_data_folder=processed_data_folder, base_dir=base_dir, n_sample_points=n_sample_points,
                     resolution=resolution, mask_resolution=mask_resolution, points_3d=points_3d, point_levels_3d=point_levels_3d,
                     path_test_dataset=path_test_dataset, path_val_dataset=path_val_dataset, path_train_dataset=path_train_dataset,
                     return_electrodes=return_electrodes)

    train_dataset = torch.load(path_train_dataset)
    val_dataset = torch.load(path_val_dataset)
    test_dataset = torch.load(path_test_dataset)
    train_dataset.apply_rotation = apply_rotation
    train_dataset.apply_subsampling = apply_subsampling
    return train_dataset, val_dataset, test_dataset

class EITData3D(Dataset):
    def __init__(self, cases, training=False, resolution=512, n_sample_points=1000,
                 train_mean=0, train_std=1, return_electrodes=True, apply_rotation=False, apply_subsampling=False,
                 min_coords=None, max_coords=None, points_3d=True, point_levels_3d=9, processed_path='data/processed/3d',
                 base_dir=''):
        self.resolution = resolution
        # use only the middle slice if 3d points are processed
        self.n_sample_points = n_sample_points
        self.training = training
        self.return_electrodes = return_electrodes
        self.cases = cases
        self.train_mean = train_mean
        self.train_std = train_std
        self.points_max = max_coords
        self.points_min = min_coords
        self.apply_rotation = apply_rotation
        self.apply_subsampling = apply_subsampling
        self.points_3d = points_3d
        self.processed_path = processed_path
        self.base_dir = base_dir
        case_files = {case: os.listdir(os.path.join(self.base_dir, self.processed_path, case)) for case in self.cases}
        self.case_files = [f"{dir_path}/{file}" for dir_path, files in case_files.items() for file in files]

    def __len__(self):
        return len(self.case_files)

    def _random_rotation_matrix(self, rotation_angle):
        rotation_angle = torch.tensor(rotation_angle)
        rotation_matrix = torch.tensor([[torch.cos(rotation_angle), -torch.sin(rotation_angle)],
                                        [torch.sin(rotation_angle), torch.cos(rotation_angle)]])
        return rotation_matrix.float()

    def _random_rotation_mask(self, mask, rotation_angle):
        # Apply the rotation
        rotated_mask = TF.rotate(mask.unsqueeze(0), rotation_angle)
        return rotated_mask.squeeze()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        case_file = self.case_files[idx]
        case_path = os.path.join(self.base_dir, self.processed_path, case_file)
        file = np.load(case_path)
        target = torch.from_numpy(file['targets'].reshape(-1, 1))
        signal =  torch.from_numpy(file['signals'])
        signal = (signal - self.train_mean) / self.train_std
        mask =  torch.from_numpy(file['masks'])
        electrode =  torch.from_numpy(file['electrodes'])
        electrode = (electrode - self.points_min) / (self.points_max - self.points_min) * 2 - 1
        points =  torch.from_numpy(file['points']).reshape(-1, 3)
        points = (points - self.points_min) / (self.points_max - self.points_min) * 2 - 1
        # if not self_use_3d:
        #     points = self.points[idx][]
        if self.training and self.apply_subsampling:
            sample_indices = torch.multinomial(torch.ones(target.flatten().shape).float(), self.n_sample_points, replacement=False)

        else:
            sample_indices = torch.arange(target.shape[0])
            
        points = points[sample_indices].float()
        target = target[sample_indices]

        if self.apply_rotation:
            if self.training and self.return_electrodes:
                rotation_angle = np.random.uniform(0, 2 * np.pi)
                rotation_matrix = self._random_rotation_matrix(rotation_angle=rotation_angle)
                rot_points = points.float().clone()
                rot_points[:,:2] = torch.matmul(rot_points[:,:2], rotation_matrix)
                rot_electrode = electrode.clone()
                rot_electrode[:,:,:,:,:2] = torch.matmul(rot_electrode[:,:,:,:,:2].float(), rotation_matrix)
                return rot_points, signal.float(), rot_electrode, mask, target.float()
        return points.float(), signal.float(), electrode, mask, target.float()

def generate_points(resolution, no_weights=False):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    yv, xv = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xv.flatten(), yv.flatten()], 1)
    return points