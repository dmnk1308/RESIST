import os
import sys
sys.path.append('../..')
sys.path.append('../')
sys.path.append('./')
from data_processing.dataset import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf
import fnmatch

def get_TCIA_cases(cfg: DictConfig, base_dir=".."):
    if cfg.data.cases == 'all':
        cases = os.listdir(os.path.join(base_dir,cfg.data.raw_data_folder))
        cases = [case for case in cases if fnmatch.fnmatch(case, 'case_TCIA_*')]
        cases_number = [int(case.split('_')[-2]) for case in cases]
        cases 
    else:
        cases = cfg.data.cases
    return cases[35:]

@hydra.main(config_path="../cfg", config_name="config")
def main(cfg: DictConfig):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    cases = get_TCIA_cases(cfg, base_dir=base_dir)

    load_dataset(cases, write=True,
                 base_dir = base_dir,
                 raw_data_folder=cfg.data.raw_data_folder, 
                 processed_data_folder=cfg.data.processed_data_folder,
                 resolution=cfg.data.resolution,
                 electrode_resolution=cfg.data.electrode_resolution)

if __name__ == '__main__':
    main()