import torch
from torch.utils.data import Dataset
from data_processing.helper import cosine_similarity, change_rho, combine_electrode_positions
import os
import fnmatch
from tqdm import tqdm
import cv2
import numpy as np
from data_processing.preprocessing import load_mask_target_electrodes, load_signals
import torchvision.transforms.functional as TF
from utils.helper import set_seeds

def write_npz_case(case, raw_data_folder="data/raw", processed_data_folder="data/processed", resolution=128, electrode_resolution=512):
    # how many levels and how many measurements per level do we have?
    if isinstance(case, int):
        case = 'case_'+str(case)
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

    # load data
    signal = load_signals(dir=case_dir, return_square=False)     
    mask, target, electrode, points = load_mask_target_electrodes(dir=case_dir,
                                                            resolution=resolution, electrode_resolution=electrode_resolution)
    # assign electrodes the right level (up - down)
    level = np.mean(electrode[:,:,-1], axis=1)
    max_level, min_level = np.max(electrode[:,:,-1]), np.min(electrode[:,:,-1])
    electrode[:,:,-1] = ((electrode[:,:,-1] - min_level) / (max_level - min_level)) * 2 - 1
    level = np.argsort(level)
    
    # copy mask and target according to the number of simulations per level
    mask = mask.repeat(level_counts, axis=0)
    target = target.repeat(level_counts, axis=0)
    target = change_rho(target, case_signals_rhos)
    electrode = electrode.repeat(level_counts, axis=0)
    level = level.repeat(level_counts, axis=0)
    points = points.repeat(level_counts, axis=0)
    # save to file
    np.savez_compressed(case_dir_processed, signals=signal, targets=target, masks=mask, electrodes=electrode, levels=level, points=points)

def make_npz(cases, raw_data_folder="data/raw", processed_data_folder="data/processed", resolution=512, electrode_resolution=512, override_npz=False):
    for case in tqdm(cases):
        if os.path.exists(os.path.join(processed_data_folder, case+'.npz')) and not override_npz:
            continue
        try:
            write_npz_case(case, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                           resolution=resolution, electrode_resolution=electrode_resolution)
        except Exception as e:
            print(case, 'can not be processed to .npz file.')
            print(e)

def make_dataset(cases, processed_data_folder="data/processed", resolution=128, electrode_resolution=512, mask_resolution=512, 
                 no_weights=False, n_sample_points=10000, return_electrodes=True, use_epair_center=False,
                 path_test_dataset="data/datasets/test_dataset.pt", path_val_dataset="data/datasets/val_dataset.pt", 
                 path_train_dataset="data/datasets/train_dataset.pt"):
    set_seeds(123)
    # load .npz files
    signals = [] 
    targets = []
    masks = []
    electrodes = []
    levels = []
    points = []
    for case in tqdm(cases):
        file_path = os.path.join(processed_data_folder, case+'.npz')
        file = np.load(file_path)
        signal = file['signals']
        target = file['targets']
        electrode = file['electrodes']
        mask = file['masks']
        level = file['levels']
        point = file['points']
        mask = np.array([cv2.resize(m.squeeze(), (mask_resolution, mask_resolution), interpolation=cv2.INTER_NEAREST) for m in mask])

        electrodes.append(torch.from_numpy(electrode))
        signals.append(torch.from_numpy(signal))
        targets.append(torch.from_numpy(target))
        masks.append(torch.from_numpy(mask))
        levels.append(torch.from_numpy(level))
        points.append(torch.from_numpy(point))
    
    # split into training, validation and test
    number_cases = len(cases)
    number_training_cases = int(number_cases*0.8)
    number_validation_cases = int(number_cases*0.1)
    number_test_cases = number_cases - number_training_cases - number_validation_cases

    # get point mean and std
    # max_coord = torch.cat(points, axis=0).numpy().max(axis=(0,1))
    # min_coord = torch.cat(points, axis=0).numpy().min(axis=(0,1))
    max_coord = torch.cat(points, axis=0).numpy()[:,:,:2].max()
    min_coord = torch.cat(points, axis=0).numpy()[:,:,:2].min()

    train_cases = cases[:number_training_cases]
    train_signals = torch.cat(signals[:number_training_cases], axis=0)
    train_targets = torch.cat(targets[:number_training_cases], axis=0)
    train_masks = torch.cat(masks[:number_training_cases], axis=0)
    train_electrodes = torch.cat(electrodes[:number_training_cases], axis=0)
    train_levels = torch.cat(levels[:number_training_cases], axis=0)
    train_points = torch.cat(points[:number_training_cases], axis=0)
    val_cases = cases[number_training_cases:number_training_cases+number_validation_cases]
    val_signals = torch.cat(signals[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_targets = torch.cat(targets[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_masks = torch.cat(masks[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_electrodes = torch.cat(electrodes[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_levels = torch.cat(levels[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_points = torch.cat(points[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    test_cases = cases[number_training_cases+number_validation_cases:]
    test_signals = torch.cat(signals[number_training_cases+number_validation_cases:], axis=0)
    test_targets = torch.cat(targets[number_training_cases+number_validation_cases:], axis=0)
    test_masks = torch.cat(masks[number_training_cases+number_validation_cases:], axis=0)
    test_electrodes = torch.cat(electrodes[number_training_cases+number_validation_cases:], axis=0)
    test_levels = torch.cat(levels[number_training_cases+number_validation_cases:], axis=0)
    test_points = torch.cat(points[number_training_cases+number_validation_cases:], axis=0)

    # normalize signals
    train_signal_mean = train_signals.mean(dim=(0,1), keepdim=True)
    train_signal_std = train_signals.std(dim=(0,1), keepdim=True)
    # train_signal_mean = train_signals.mean()
    # train_signal_std = train_signals.std()
    train_signals = (train_signals - train_signal_mean) / train_signal_std
    test_signals = (test_signals - train_signal_mean) / train_signal_std
    val_signals = (val_signals - train_signal_mean) / train_signal_std
    # normalize points and electrode position
    train_points = ((train_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    val_points = ((val_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    test_points = ((test_points - min_coord) / (max_coord - min_coord)) * 2 - 1
    # train_electrodes[:,:,:2] = ((train_electrodes[:,:,:2] - min_coord[:2]) / (max_coord[:2] - min_coord[:2])) * 2 - 1
    # val_electrodes[:,:,:2] = ((val_electrodes[:,:,:2] - min_coord[:2]) / (max_coord[:2] - min_coord[:2])) * 2 - 1
    # test_electrodes[:,:,:2] = ((test_electrodes[:,:,:2] - min_coord[:2]) / (max_coord[:2] - min_coord[:2])) * 2 - 1
    train_electrodes[:,:,:2] = ((train_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1
    val_electrodes[:,:,:2] = ((val_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1
    test_electrodes[:,:,:2] = ((test_electrodes[:,:,:2] - min_coord) / (max_coord - min_coord)) * 2 - 1


    train_dataset = EITData(train_signals, train_targets, train_masks, train_electrodes, train_levels, train_cases, train_points, resolution=resolution, training=True, 
                            no_weights=no_weights, n_sample_points=n_sample_points, train_mean=train_signal_mean, train_std=train_signal_std,
                            return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord, use_epair_center=use_epair_center)
    val_dataset = EITData(val_signals, val_targets, val_masks, val_electrodes, val_levels, val_cases, val_points, resolution=resolution, training=False, no_weights=no_weights, 
                          train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord, use_epair_center=use_epair_center)
    test_dataset = EITData(test_signals, test_targets, test_masks, test_electrodes, test_levels, test_cases, test_points, resolution=resolution, training=False, no_weights=no_weights, 
                           train_mean=train_signal_mean, train_std=train_signal_std, return_electrodes=return_electrodes, min_coords=min_coord, max_coords=max_coord, use_epair_center=use_epair_center)
    print(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}')

    torch.save(train_dataset, path_train_dataset)
    torch.save(val_dataset, path_val_dataset)
    torch.save(test_dataset, path_test_dataset)


def load_dataset(cases, resolution=128, electrode_resolution=512, mask_resolution=512, n_sample_points=10000,
                 base_dir="..", raw_data_folder="data/raw", processed_data_folder="data/processed", dataset_data_folder="data/datasets", 
                 no_weights=False, name_prefix="", write_dataset=False, write_npz=False, overwrite_npz=False, return_electrodes=True,
                 apply_rotation=False, apply_subsampling=True, use_epair_center=False):
    # set up paths
    raw_data_folder = os.path.normpath(os.path.join(base_dir,raw_data_folder))
    processed_data_folder = os.path.normpath(os.path.join(base_dir,processed_data_folder))
    dataset_data_folder = os.path.normpath(os.path.join(base_dir,dataset_data_folder))        
    path_train_dataset = os.path.join(dataset_data_folder,'train_dataset'+name_prefix+'.pt')
    path_val_dataset = os.path.join(dataset_data_folder,'val_dataset'+name_prefix+'.pt')
    path_test_dataset = os.path.join(dataset_data_folder,'test_dataset'+name_prefix+'.pt')

    if write_npz:
        make_npz(cases, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                 resolution=resolution, electrode_resolution=electrode_resolution, override_npz=overwrite_npz)
    if write_dataset:
        make_dataset(cases, processed_data_folder=processed_data_folder, n_sample_points=n_sample_points,
                     resolution=resolution, electrode_resolution=electrode_resolution, mask_resolution=mask_resolution, no_weights=no_weights, 
                     path_test_dataset=path_test_dataset, path_val_dataset=path_val_dataset, path_train_dataset=path_train_dataset,
                     return_electrodes=return_electrodes, use_epair_center=use_epair_center)

    train_dataset = torch.load(path_train_dataset)
    val_dataset = torch.load(path_val_dataset)
    test_dataset = torch.load(path_test_dataset)
    train_dataset.apply_rotation = apply_rotation
    train_dataset.apply_subsampling = apply_subsampling
    train_dataset.use_epair_center, val_dataset.use_epair_center, test_dataset.use_epair_center = [use_epair_center]*3
    return train_dataset, val_dataset, test_dataset
    
class EITData(Dataset):
    def __init__(self, signals, targets, masks, electrodes, levels, cases, points, training=False, resolution=128, n_sample_points=1000,
                 no_weights=False, train_mean=0, train_std=1, return_electrodes=True, apply_rotation=False, apply_subsampling=False,
                 min_coords=None, use_epair_center=False, max_coords=None):
        self.no_weights = no_weights
        self.resolution = resolution
        self.signals = signals
        self.targets = targets
        self.masks = masks
        self.points = points
        self.n_sample_points = n_sample_points
        self.training = training
        self.no_weights = no_weights
        self.return_electrodes = return_electrodes
        self.cases = cases
        self.levels = levels
        self.train_mean = train_mean
        self.train_std = train_std
        self.points_max = max_coords
        self.points_min = min_coords
        self.apply_rotation = apply_rotation
        self.apply_subsampling = apply_subsampling
        self.use_epair_center = use_epair_center
        
        epair_centers = []
        # weights = []
        # points = generate_points(resolution=self.resolution)
        for electrode in electrodes:
            epair_center = get_epair_center(electrode)
            epair_centers.append(epair_center)
        #     if no_weights == False:
        #         w = get_weights(epair_center, points)
        #     else:
        #         w = torch.ones((1,1))
        #     weights.append(w)
        epair_centers = torch.stack(epair_centers, 0)

        electrodes = combine_electrode_positions(electrodes.float(), use_epair_center=False) 
        epair_centers = combine_electrode_positions(epair_centers, use_epair_center=True)
        self.epair_center = epair_centers
        self.electrodes = electrodes

        # self.weights = weights

    def __len__(self):
        return self.targets.shape[0]

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
        
        target = self.targets[idx]
        signal = self.signals[idx]
        mask = self.masks[idx]
        if self.use_epair_center:
            electrode = self.epair_center[idx]
        else:
            electrode = self.electrodes[idx]
        points = self.points[idx][:,:2]

        if self.training and self.apply_subsampling:
            sample_indices = torch.multinomial(torch.ones(self.resolution**2).float(), self.n_sample_points, replacement=False)

        else:
            sample_indices = torch.arange(self.resolution**2)
            
        # points = self.points
        if self.no_weights:
            weights = torch.ones((1,1))
        else:
            weights = self.weights[idx]
            weights = weights[sample_indices].float()
        points = points[sample_indices].float()

        target = target.reshape(self.resolution**2, 1)[sample_indices]

        if self.apply_rotation:
            if self.training and self.return_electrodes:
                rotation_angle = np.random.uniform(0, 2 * np.pi)
                rotation_matrix = self._random_rotation_matrix(rotation_angle=rotation_angle)
                # extreme_points = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).float()
                # extreme_points = torch.matmul(extreme_points, rotation_matrix.T)
                # x_max, x_min = extreme_points[:,0].max(), extreme_points[:,0].min()
                # y_max, y_min = extreme_points[:,1].max(), extreme_points[:,1].min()
                rot_points = torch.matmul(points.float().clone(), rotation_matrix)
                # points = torch.matmul(points, rotation_matrix)
                rot_electrode = electrode.clone()
                rot_electrode[:,:,:,:2] = torch.matmul(rot_electrode[:,:,:,:2].float(), rotation_matrix)
                # electrode[:,:,:,:2] = torch.matmul(electrode[:,:,:,:2], rotation_matrix)
                # rescale
                # points = ((points - torch.tensor([[x_min, y_min]])) / (torch.tensor([[x_max, y_max]]) - torch.tensor([[x_min, y_min]]))) * 2 - 1
                # electrode[:,:2] = ((electrode[:,:2] - torch.tensor([[x_min, y_min]])) / (torch.tensor([[x_max, y_max]]) - torch.tensor([[x_min, y_min]]))) * 2 - 1
                # add location noise
                # noise = (torch.rand_like(points) - 0.5)/self.resolution
                # points = points + noise
                # mask = self._random_rotation_mask(mask=mask, rotation_angle=rotation_angle)
                return rot_points, weights, signal.float(), rot_electrode, mask, target.float()
         
        if not self.return_electrodes:
            electrode = self.levels[idx]

        # signal = signal + torch.randn_like(signal)
        # signal = torch.randn_like(signal)

        return points.float(), weights, signal.float(), electrode, mask, target.float()

def get_epair_center(electrodes):
    epair_centers = []
    for i in range(16):
        if i == 15:
            j = 0
        else:
            j = i+1
        epair_center = (electrodes[i] + electrodes[j])/2
        epair_centers.append(epair_center.float())
    return torch.stack(epair_centers, 0)


def generate_points(resolution, no_weights=False):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    yv, xv = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xv.flatten(), yv.flatten()], 1)
    return points