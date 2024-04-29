import os
import sys
import torch
sys.path.append('../..')
sys.path.append('../')
from data_processing.dataset import load_dataset
from train.training import training
from train.testing import testing
from utils.helper import get_all_cases
import hydra
from omegaconf import DictConfig, OmegaConf
import fnmatch
import random
import wandb
import argparse
from utils.helper import set_seeds

@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg: DictConfig, inference=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = '../'
    run = wandb.init(project='mask_segmentation', 
                     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    cases = get_all_cases(cfg, base_dir='../')
    train_dataset, val_dataset, test_dataset = load_dataset(cases,
                resolution=cfg.data.resolution, 
                electrode_resolution=cfg.data.electrode_resolution,
                mask_resolution=cfg.data.mask_resolution, 
                base_dir = script_dir,
                raw_data_folder=cfg.data.raw_data_folder, 
                processed_data_folder=cfg.data.processed_data_folder,
                dataset_data_folder=cfg.data.dataset_data_folder,
                no_weights=cfg.data.no_weights, name_prefix=cfg.data.name_prefix,
                write_dataset=cfg.data.write_dataset, write_npz=cfg.data.write_npz, 
                overwrite_npz=cfg.data.overwrite_npz, n_sample_points=cfg.learning.training.sample_points,
                return_electrodes=cfg.data.return_electrodes, apply_rotation=cfg.data.apply_rotation,
                apply_subsampling=cfg.data.apply_subsampling)
    model = hydra.utils.instantiate(cfg.learning.segmentation_model)


if __name__ == '__main__':
    set_seeds(12)
    main()

