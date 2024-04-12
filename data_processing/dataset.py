import torch
from torch.utils.data import Dataset
from data_processing.helper import cosine_similarity, change_rho
import os
import fnmatch
from tqdm import tqdm
import cv2
import numpy as np
from data_processing.obj2py import read_egt, read_get
from data_processing.preprocessing import load_mask_target_electrodes, load_tomograms, load_signals, write_data
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF

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
    signal = load_signals(dir=case_dir)     
    mask, target, electrode = load_mask_target_electrodes(dir=case_dir,
                                                            resolution=resolution, electrode_resolution=electrode_resolution)
    # copy mask and target according to the number of simulations per level
    mask = mask.repeat(level_counts, axis=0)
    target = target.repeat(level_counts, axis=0)
    target = change_rho(target, case_signals_rhos)
    electrode = electrode.repeat(level_counts, axis=0)
    # save to file
    np.savez_compressed(case_dir_processed, signals=signal, targets=target, masks=mask, electrodes=electrode)

def make_npz(cases, raw_data_folder="data/raw", processed_data_folder="data/processed", resolution=512, electrode_resolution=512, override_npz=False):
    for case in tqdm(cases):
        if os.path.exists(os.path.join(processed_data_folder, case+'.npz')) and not override_npz:
            continue
        try:
            write_npz_case(case, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, resolution=resolution, electrode_resolution=electrode_resolution)
        except Exception as e:
            print(case, 'can not be processed to .npz file.')
            print(e)

def make_dataset(cases, processed_data_folder="data/processed", resolution=128, electrode_resolution=512, mask_resolution=512, no_weights=False, n_sample_points=10000,
                 path_test_dataset="data/datasets/test_dataset.pt", path_val_dataset="data/datasets/val_dataset.pt", path_train_dataset="data/datasets/train_dataset.pt"):
    # load .npz files
    signals = [] 
    targets = []
    masks = []
    electrodes = []
    for case in tqdm(cases):
        file_path = os.path.join(processed_data_folder, case+'.npz')
        file = np.load(file_path)
        signal = file['signals']
        target = file['targets']
        electrode = file['electrodes']
        mask = file['masks']
        mask = np.array([cv2.resize(m.squeeze(), (mask_resolution, mask_resolution), interpolation=cv2.INTER_NEAREST) for m in mask])
        electrodes.append(torch.from_numpy(electrode))
        signals.append(torch.from_numpy(signal))
        targets.append(torch.from_numpy(target))
        masks.append(torch.from_numpy(mask))
    
    # split into training, validation and test
    number_cases = len(cases)
    number_training_cases = int(number_cases*0.8)
    number_validation_cases = int(number_cases*0.1)
    number_test_cases = number_cases - number_training_cases - number_validation_cases

    train_signals = torch.cat(signals[:number_training_cases], axis=0)
    train_targets = torch.cat(targets[:number_training_cases], axis=0)
    train_masks = torch.cat(masks[:number_training_cases], axis=0)
    train_electrodes = torch.cat(electrodes[:number_training_cases], axis=0)
    val_signals = torch.cat(signals[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_targets = torch.cat(targets[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_masks = torch.cat(masks[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    val_electrodes = torch.cat(electrodes[number_training_cases:number_training_cases+number_validation_cases], axis=0)
    test_signals = torch.cat(signals[number_training_cases+number_validation_cases:], axis=0)
    test_targets = torch.cat(targets[number_training_cases+number_validation_cases:], axis=0)
    test_masks = torch.cat(masks[number_training_cases+number_validation_cases:], axis=0)
    test_electrodes = torch.cat(electrodes[number_training_cases+number_validation_cases:], axis=0)

    train_signal_mean = train_signals.mean()
    train_signal_std = train_signals.std()
    train_signals = (train_signals - train_signal_mean) / train_signal_std
    test_signals = (test_signals - train_signal_mean) / train_signal_std
    val_signals = (val_signals - train_signal_mean) / train_signal_std

    train_dataset = EITData(train_signals, train_targets, train_masks, train_electrodes, resolution=resolution, training=True, 
                            no_weights=no_weights, n_sample_points=n_sample_points, train_mean=train_signal_mean, train_std=train_signal_std)
    val_dataset = EITData(val_signals, val_targets, val_masks, val_electrodes, resolution=resolution, training=False, no_weights=no_weights, 
                          train_mean=train_signal_mean, train_std=train_signal_std)
    test_dataset = EITData(test_signals, test_targets, test_masks, test_electrodes, resolution=resolution, training=False, no_weights=no_weights, 
                           train_mean=train_signal_mean, train_std=train_signal_std)
    print(f'Training set: {len(train_dataset)}, validation set: {len(val_dataset)}, test set: {len(test_dataset)}')

    torch.save(train_dataset, path_train_dataset)
    torch.save(val_dataset, path_val_dataset)
    torch.save(test_dataset, path_test_dataset)


def load_dataset(cases, resolution=128, electrode_resolution=512, mask_resolution=512, n_sample_points=10000,
                 base_dir="..", raw_data_folder="data/raw", processed_data_folder="data/processed", dataset_data_folder="data/datasets", 
                 no_weights=False, name_prefix="", write_dataset=False, write_npz=False):
    # set up paths
    raw_data_folder = os.path.normpath(os.path.join(base_dir,raw_data_folder))
    processed_data_folder = os.path.normpath(os.path.join(base_dir,processed_data_folder))
    dataset_data_folder = os.path.normpath(os.path.join(base_dir,dataset_data_folder))        
    path_train_dataset = os.path.join(dataset_data_folder,'train_dataset'+name_prefix+'.pt')
    path_val_dataset = os.path.join(dataset_data_folder,'val_dataset'+name_prefix+'.pt')
    path_test_dataset = os.path.join(dataset_data_folder,'test_dataset'+name_prefix+'.pt')

    if write_npz:
        make_npz(cases, raw_data_folder=raw_data_folder, processed_data_folder=processed_data_folder, 
                 resolution=resolution, electrode_resolution=electrode_resolution)
    if write_dataset:
        make_dataset(cases, processed_data_folder=processed_data_folder, n_sample_points=n_sample_points,
                     resolution=resolution, electrode_resolution=electrode_resolution, mask_resolution=mask_resolution, no_weights=no_weights, 
                     path_test_dataset=path_test_dataset, path_val_dataset=path_val_dataset, path_train_dataset=path_train_dataset)

    train_dataset = torch.load(path_train_dataset)
    val_dataset = torch.load(path_val_dataset)
    test_dataset = torch.load(path_test_dataset)
    return train_dataset, val_dataset, test_dataset
    
class EITData(Dataset):
    def __init__(self, signals, targets, masks, electrodes, transform=None, training=True, resolution=128, n_sample_points=1000,
                 no_weights=False, train_mean=0, train_std=1):
        self.no_weights = no_weights
        self.resolution = resolution
        self.signals = signals
        self.targets = targets
        self.masks = masks
        electrodes[:,:,:2] = (electrodes[:,:,:2] / self.resolution) * 2 - 1
        self.electrodes = electrodes#[:,:,:2]
        self.n_sample_points = n_sample_points
        self.training = training
        self.transform = transform
        self.no_weights = no_weights

        impulses = []
        weights = []
        points = generate_points(resolution=self.resolution)
        for electrode in self.electrodes:
            impulse = get_impulses(electrode[:,:2])
            impulses.append(impulse)
            if no_weights == False:
                w = get_weights(impulse, points)
            else:
                w = torch.ones((1,1))
            weights.append(w)
        self.impulses = impulses
        self.points = points
        self.weights = weights

    def __len__(self):
        return self.targets.shape[0]

    def _random_rotation_matrix(self, rotation_angle):
        rotation_matrix = torch.tensor([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                        [np.sin(rotation_angle), np.cos(rotation_angle)]])
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
        electrode = self.electrodes[idx]
        impulse = self.impulses[idx]

        if self.training:
            sample_indices = torch.multinomial(torch.ones(self.resolution**2).float(), self.n_sample_points, replacement=False)

        else:
            sample_indices = torch.arange(self.resolution**2)
            
        points = self.points
        if self.no_weights:
            weights = torch.ones((1,1))
        else:
            weights = self.weights[idx]
            weights = weights[sample_indices].float()
        points = points[sample_indices]

        electrode = torch.cat((impulse, electrode[:,2].unsqueeze(-1)), 1).float()
        target = target.reshape(self.resolution**2, 1)[sample_indices]
        sample = [signal, mask.unsqueeze(0), electrode, points, target]

        if self.transform:
            sample = self.transform(sample)
        
        if self.training:
            rotation_angle = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = self._random_rotation_matrix(rotation_angle=rotation_angle)
            points = torch.matmul(points, rotation_matrix.T)
            electrode[:,:2] = torch.matmul(electrode[:,:2], rotation_matrix.T)
            mask = self._random_rotation_mask(mask=mask, rotation_angle=rotation_angle)
        
        # signal = signal + torch.randn_like(signal)
        # signal = torch.randn_like(signal)

        return points.float(), weights, signal.float(), electrode, mask, target.float()

def get_impulses(electrodes):
    impulses = []
    for i in range(16):
        if i == 15:
            j = 0
        else:
            j = i+1
        impulse = (electrodes[i] + electrodes[j])/2
        impulses.append(impulse.float())
    return torch.stack(impulses, 0)

def get_positional_weights(impulses, points, norm=1):
    dist = -torch.cdist(impulses.unsqueeze(0), points.unsqueeze(0), p=norm)
    pos_weight = torch.nn.functional.softmax(dist.T, 1)
    return pos_weight

def get_angle_weights(impulses, points):
    ang_weights = torch.zeros((points.shape[0], 16, 16))
    for i, ref_impulse in enumerate(impulses):
        impulses_tmp = impulses - ref_impulse
        points_tmp = points - ref_impulse
        cosine_sim = cosine_similarity(impulses_tmp, points_tmp)
        cosine_sim[cosine_sim<0] = 0.
        ang_weights[:, i] = cosine_sim
    return ang_weights

def get_weights(impulses, points, norm=1):
    angle_weights = get_angle_weights(impulses, points)
    positional_weights = get_positional_weights(impulses, points)
    #weights = torch.nn.functional.softmax(aw + pw, 0)
    positional_weights = positional_weights.tile(1,1,16)
    positional_weights = torch.transpose(positional_weights, 1, 2)
    weights = angle_weights * positional_weights
    return weights

def generate_points(resolution, no_weights=False):
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    yv, xv = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([xv.flatten(), yv.flatten()], 1)
    # if no_weights == False:
    #     # angle_weights = get_angle_weights(impulses, points)
    #     # positional_weights = get_positional_weights(impulses, points)
    #     weights = get_weights(impulses, points)
    return points