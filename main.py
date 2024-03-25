import os
import sys
import torch
sys.path.append('../..')
from data_processing.dataset import load_dataset
from train.training import training
from train.testing import testing
import hydra
from omegaconf import DictConfig, OmegaConf
import fnmatch
import random
import wandb
import argparse
random.seed(123)
torch.random.manual_seed(123)


def get_all_cases(cfg: DictConfig, base_dir=".."):
    if cfg.data.cases == 'all':
        cases = os.listdir(os.path.join(base_dir,cfg.data.raw_data_folder))
        cases = [case.split('.')[0] for case in cases if fnmatch.fnmatch(case, 'case_TCIA*')]
        cases_number = [int(case.split('_')[-2]) for case in cases]
        # cases = [case for case, case_number in zip(cases, cases_number) if case_number < 190]
        # cases 
    else:
        cases = cfg.data.cases
    return cases

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig, inference=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    run = wandb.init(project=cfg.wandb.project, 
                     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    cases = get_all_cases(cfg, base_dir=script_dir)
    train_dataset, test_dataset, val_datset = load_dataset(cases,
                resolution=cfg.data.resolution, 
                electrode_resolution=cfg.data.electrode_resolution,
                mask_resolution=cfg.data.mask_resolution, 
                base_dir = script_dir,
                raw_data_folder=cfg.data.raw_data_folder, 
                processed_data_folder=cfg.data.processed_data_folder,
                dataset_data_folder=cfg.data.dataset_data_folder,
                no_weights=cfg.data.no_weights, name_prefix=cfg.data.name_prefix,
                write_dataset=cfg.data.write_dataset, write_npz=cfg.data.write_npz)

    model = hydra.utils.instantiate(cfg.learning.model, mask_resolution=cfg.data.mask_resolution, no_weights=cfg.data.no_weights)

    if not cfg.inference:
        model = training(model, train_dataset, val_datset, epochs=cfg.learning.training.epochs, 
                batch_size_train=cfg.learning.training.batch_size_train, 
                batch_size_val=cfg.learning.training.batch_size_val, lr=cfg.learning.training.learning_rate, device=cfg.learning.training.device)
    else:
        model.load_state_dict(torch.load(os.path.join(cfg.inference_path,'model.pt')))
    testing(model, test_dataset, batch_size=cfg.learning.testing.batch_size_test, device=cfg.learning.training.device)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--inference', action='store_true')
    # args = parser.parse_args()
    main()

