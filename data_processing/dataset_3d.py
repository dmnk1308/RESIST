import torch
from torch.utils.data import Dataset
from data_processing.helper import change_rho, combine_electrode_positions
import os
import fnmatch
from tqdm import tqdm
import cv2
import numpy as np
from data_processing.preprocessing import load_signals, load_body_shape, load_targets_electrodes_points
from data_processing.helper import sort_filenames
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
from utils.helper import set_seeds
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

def load_dataset_3d(cases, resolution=512, n_sample_points=10000,
                 base_dir="", raw_data_folder="data/raw", processed_data_folder="data/processed", dataset_data_folder="data/datasets", 
                 name_prefix="", write_dataset=False, write_npz=False, overwrite_npz=False, return_electrodes=True,
                 apply_rotation=False, apply_subsampling=True, apply_translation=True, translation_x=0, translation_y=0, translation_z=0,
                 point_levels_3d=9, point_range_3d=0.05, num_workers=None, multi_process=True, all_signals=False,
                 use_body_mask=False):
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
                 resolution=resolution, overwrite_npz=overwrite_npz,
                 point_levels_3d=point_levels_3d, point_range_3d=point_range_3d, all_signals=all_signals,
                 num_workers=num_workers)
        else:
            make_npz_3d(cases, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                 resolution=resolution, overwrite_npz=overwrite_npz,
                 point_levels_3d=point_levels_3d, point_range_3d=point_range_3d, all_signals=all_signals)
    if write_dataset:
        make_dataset_3d(cases, processed_data_folder=processed_data_folder, base_dir=base_dir, n_sample_points=n_sample_points,
                     resolution=resolution, point_levels_3d=point_levels_3d,
                     apply_subsampling=apply_subsampling, apply_rotation=apply_rotation, apply_translation=apply_translation,
                     translation_x=translation_x, translation_y=translation_y, translation_z=translation_z,
                     path_test_dataset=path_test_dataset, path_val_dataset=path_val_dataset, path_train_dataset=path_train_dataset,
                     return_electrodes=return_electrodes, use_body_mask=use_body_mask)

    train_dataset = torch.load(path_train_dataset)
    val_dataset = torch.load(path_val_dataset)
    test_dataset = torch.load(path_test_dataset)
    train_dataset.apply_rotation = apply_rotation
    train_dataset.apply_subsampling = apply_subsampling
    train_dataset.apply_translation = apply_translation
    train_dataset.translation_x = translation_x
    train_dataset.translation_y = translation_y
    train_dataset.translation_z = translation_z
    train_dataset.point_levels_3d, val_dataset.point_levels_3d, test_dataset.point_levels_3d = [point_levels_3d]*3
    return train_dataset, val_dataset, test_dataset

def make_npz_3d_multi(cases, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=512, overwrite_npz=False,
                point_levels_3d=9, point_range_3d=0.05, num_workers='all', all_signals=False):
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
                                         resolution=resolution, point_levels_3d=point_levels_3d, 
                                         point_range_3d=point_range_3d, multi=True, all_signals=all_signals)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(partial_function, item) for item in cases_copy]
        # Collect results with a progress bar
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
            result = future.result()
            results.append(result)

def make_npz_3d(cases, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=512, overwrite_npz=False,
                point_levels_3d=8, point_range_3d=0.05, all_signals=False):
    with tqdm(cases) as pbar:
        for case in pbar:
            pbar.set_description(f"Processing {case}.")
            if os.path.exists(os.path.join(processed_data_folder, case+'_3d.npz')) and not overwrite_npz:
                continue
            try:
                write_npz_case_3d(case, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                                resolution=resolution, all_signals=all_signals,
                                point_levels_3d=point_levels_3d, point_range_3d=point_range_3d)
            except Exception as e:
                print(case, 'can not be processed to .npz file.')
                print(e)

def write_npz_case_3d(case, raw_data_folder="data/raw", processed_data_folder="data/processed/3d", resolution=128,
                      point_levels_3d=9, point_range_3d=0.05, multi=False, all_signals=False):
    # how many levels and how many measurements per level do we have?
    print(case)
    if isinstance(case, int):
        case = 'case_'+str(case)
    case_dir = os.path.join(raw_data_folder, case)
    case_dir_processed = os.path.join(processed_data_folder, case)
    
    # load data
    signal, rhos, level = load_signals(dir=case_dir, return_square=False)  
    n_rhos = np.unique(rhos).shape[0]
    n_level = np.unique(level).shape[0]
    signal = np.moveaxis(signal.reshape(n_level, n_rhos, -1), 1, 0)
    rhos = np.moveaxis(np.array(rhos).reshape(n_level, n_rhos), 1, 0)
    level = np.moveaxis(np.array(level).reshape(n_level, n_rhos), 1, 0)
    targets, electrode, points = load_targets_electrodes_points(case_dir, resolution=resolution, all_signals=all_signals, point_levels_3d=point_levels_3d)

    # define desired shapes for all tensors
    target_shape = (n_rhos, point_levels_3d, resolution, resolution) 
    tissue_shape = (point_levels_3d, resolution, resolution)
    point_shape = (point_levels_3d, -1, 3)
    electrode_shape = (n_level, -1, 3)
    signals_shape = (n_rhos, n_level, -1)    

    rhos = rhos[:,0]
    tissue = np.copy(targets).reshape(tissue_shape)
    targets = change_rho(targets, rhos)
    points = points.reshape(point_shape)
    targets = targets.reshape(target_shape)
    signal = signal.reshape(signals_shape)
    electrode = electrode.reshape(electrode_shape)
    electrode = combine_electrode_positions(electrode) 
    
    # load body mask
    mask = load_body_shape(dir=case_dir, density=0.01)

    # make directory for each case
    os.makedirs(os.path.join(case_dir_processed), exist_ok=True)
    for s, t, r in zip(signal, targets, rhos):
        # for now only save 5, 10, 15, 20 rho
        if r in [5, 10, 15, 20]:
            np.savez_compressed(os.path.join(case_dir_processed,case+'_'+str(r)+'.npz'), signals=s, targets=t, masks=mask, electrodes=electrode, points=points, tissue=tissue)

def make_dataset_3d(cases, processed_data_folder="data/processed", base_dir='',resolution=512, 
                 n_sample_points=10000, return_electrodes=True, point_levels_3d=9,
                 apply_rotation=False, apply_subsampling=True, apply_translation=True, translation_x=0, translation_y=0, translation_z=0,
                 path_test_dataset="data/datasets/test_dataset_3d.pt", path_val_dataset="data/datasets/val_datasett_3d.pt", 
                 path_train_dataset="data/datasets/train_datasett_3d.pt", use_body_mask=False):
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
            max_coord = points[:,:,:2].max()
            min_coord = points[:,:,:2].min() 
            max_coord_z = points[:,:,2].max()
            min_coord_z = points[:,:,2].min()
        else:
            if max_coord < points[:,:,:2].max():
                max_coord = points[:,:,:2].max()
            if min_coord > points[:,:,:2].min():
                min_coord = points[:,:,:2].min()
            if max_coord_z < points[:,:,2].max():
                max_coord_z = points[:,:,2].max()
            if min_coord_z > points[:,:,2].min():
                min_coord_z = points[:,:,2].min()
        if case in train_cases:
            signals.append(np.load(file_path)['signals'])
    # normalize signals
    train_signals = torch.from_numpy(np.concatenate(signals, axis=0)).reshape(-1, 4, 16, 13)
    # each channel separetely (4, 16, 13)
    train_signal_mean = train_signals.mean(dim=(0), keepdim=True)
    train_signal_std = train_signals.std(dim=(0), keepdim=True)
    # 13 channels in cycle (1, 1, 1, 13)
    # train_signal_mean = train_signals.mean(dim=(0,1,2), keepdim=True)
    # train_signal_std = train_signals.std(dim=(0,1,2), keepdim=True)
    # across all channels (1, 1, 1, 1)
    # train_signal_mean = train_signals.mean(dim=(0,1,2,3), keepdim=True)
    # train_signal_std = train_signals.std(dim=(0,1,2,3), keepdim=True)

    train_dataset = EITData3D(train_cases, resolution=resolution, training=True, n_sample_points=n_sample_points, train_mean=train_signal_mean, train_std=train_signal_std, 
                              point_levels_3d=point_levels_3d, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord, min_coords_z=min_coord_z, max_coords_z=max_coord_z,
                              processed_path=processed_data_folder, base_dir=base_dir, apply_rotation=apply_rotation, apply_subsampling=apply_subsampling, 
                              apply_translation=apply_translation, translation_x=translation_x, translation_y=translation_y, translation_z=translation_z, 
                              use_body_mask=use_body_mask)
    val_dataset = EITData3D(val_cases, resolution=resolution, training=False, point_levels_3d=point_levels_3d,
                            train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord,min_coords_z=min_coord_z, max_coords_z=max_coord_z,
                            processed_path=processed_data_folder, base_dir=base_dir, use_body_mask=use_body_mask)
    test_dataset = EITData3D(test_cases, resolution=resolution, training=False, point_levels_3d=point_levels_3d,                             
                            train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord,min_coords_z=min_coord_z, max_coords_z=max_coord_z,
                            processed_path=processed_data_folder, base_dir=base_dir, use_body_mask=use_body_mask)
    print(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}')

    torch.save(train_dataset, path_train_dataset)
    torch.save(val_dataset, path_val_dataset)
    torch.save(test_dataset, path_test_dataset)

class EITData3D(Dataset):
    def __init__(self, cases, training=False, resolution=512, n_sample_points=1000,
                 train_mean=0, train_std=1, return_electrodes=True, apply_rotation=False, apply_subsampling=False,
                 apply_translation=False, translation_x=0.2, translation_y=0.2, translation_z=0.05, use_body_mask=False,
                 min_coords=None, max_coords=None, min_coords_z=None, max_coords_z=None, point_levels_3d=9, processed_path='data/processed/3d',
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
        self.points_max_z = max_coords_z
        self.points_min_z = min_coords_z
        self.apply_rotation = apply_rotation
        self.apply_subsampling = apply_subsampling
        self.processed_path = processed_path
        self.base_dir = base_dir
        case_files = {case: os.listdir(os.path.join(self.base_dir, self.processed_path, case)) for case in self.cases}
        case_files = [f"{dir_path}/{file}" for dir_path, files in case_files.items() for file in files]
        self.case_files = sort_filenames(case_files)
        self.point_levels_3d = point_levels_3d
        self.apply_subsampling = apply_subsampling
        self.apply_rotation = apply_rotation
        self.apply_translation = apply_translation
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z
        self.use_body_mask = use_body_mask

    def __len__(self):
        return len(self.case_files)

    def _random_rotation_matrix(self, rotation_angle):
        rotation_angle = torch.tensor(rotation_angle)
        rotation_matrix = torch.tensor([[torch.cos(rotation_angle), -torch.sin(rotation_angle)],
                                        [torch.sin(rotation_angle), torch.cos(rotation_angle)]])
        return rotation_matrix.float()
    
    def _random_translation(self, points, electrode):
        translation_x = np.random.uniform(-self.translation_x, self.translation_x, 1)
        translation_y = np.random.uniform(-self.translation_y, self.translation_y, 1)
        translation_z = np.random.uniform(-self.translation_z, self.translation_z, 1)
        translation = np.array([translation_x, translation_y, translation_z]).reshape(3)
        trans_points = points.float() + translation
        trans_electrode = electrode.clone() + translation
        return trans_points, trans_electrode

    def _random_rotation(self, points, electrode):
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = self._random_rotation_matrix(rotation_angle=rotation_angle)
        rot_points = points.float().clone()
        rot_points[:,:2] = torch.matmul(rot_points[:,:2], rotation_matrix)
        rot_electrode = electrode.clone()
        rot_electrode[:,:,:,:,:2] = torch.matmul(rot_electrode[:,:,:,:,:2].float(), rotation_matrix)
        return rot_points, rot_electrode

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
        tissue = torch.from_numpy(file['tissue'].reshape(-1, 1))
        signal =  torch.from_numpy(file['signals']).reshape(-1, 4, 16, 13)
        signal = (signal - self.train_mean) / self.train_std
        signal = signal.reshape(4, -1)
        if self.use_body_mask:
            mask =  torch.from_numpy(file['masks'])
        else:
            mask = torch.zeros((1,1))
        electrode =  torch.from_numpy(file['electrodes'])
        electrode[:,:,:,:,:2] = (electrode[:,:,:,:,:2] - self.points_min) / (self.points_max - self.points_min) * 2 - 1
        electrode[:,:,:,:,2] = (electrode[:,:,:,:,2] - self.points_min_z) / (self.points_max_z - self.points_min_z) * 2 - 1
        points =  torch.from_numpy(file['points']).reshape(-1, 3)
        points[:,:2] = (points[:,:2] - self.points_min) / (self.points_max - self.points_min) * 2 - 1
        points[:,2] = (points[:,2] - self.points_min_z) / (self.points_max_z - self.points_min_z) * 2 - 1

        if self.training:
            if self.apply_rotation:
                points, electrode = self._random_rotation(points, electrode)
            if self.apply_translation:
                points, electrode = self._random_translation(points, electrode)
            # # filter out points that are outside of [-1,1]
            # filter_idx = torch.any(torch.logical_or(points > 1, points < -1), dim=1)
            # points = points[~filter_idx]
            # target = target[~filter_idx]
            # tissue = tissue[~filter_idx]
            if self.apply_subsampling:
                sample_indices = torch.multinomial(torch.ones(target.flatten().shape).float(), self.n_sample_points, replacement=False)
            else:
                sample_indices = torch.arange(target.shape[0])
            points = points[sample_indices].float()
            target = target[sample_indices]
            tissue = tissue[sample_indices]
        return points.float(), signal.float(), electrode, mask, target.float(), tissue

def generate_points(resolution):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    yv, xv = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xv.flatten(), yv.flatten()], 1)
    return points