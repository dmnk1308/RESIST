import os
import sys
import torch
sys.path.append('../..')
from data_processing.dataset_3d import load_dataset_3d
from train.training import training
from train.testing import testing
from utils.helper import get_all_cases, set_seeds
from segmentation.training import training as seg_training
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import argparse

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig, inference=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.chdir(script_dir)
    cases = get_all_cases(cfg, base_dir=script_dir, use_raw=True)

    train_dataset, val_dataset, test_dataset = load_dataset_3d(cases,
                resolution=cfg.data.resolution, 
                electrode_resolution=cfg.data.electrode_resolution,
                mask_resolution=cfg.data.mask_resolution, 
                base_dir = script_dir,
                raw_data_folder=cfg.data.raw_data_folder, 
                processed_data_folder=cfg.data.processed_data_folder,
                dataset_data_folder=cfg.data.dataset_data_folder,
                name_prefix=cfg.data.name_prefix,
                write_dataset=cfg.data.write_dataset, write_npz=cfg.data.write_npz, 
                overwrite_npz=cfg.data.overwrite_npz, n_sample_points=cfg.learning.training.sample_points,
                return_electrodes=cfg.data.return_electrodes, apply_rotation=cfg.data.apply_rotation,
                apply_subsampling=cfg.data.apply_subsampling,
                apply_translation = cfg.data.apply_translation,
                translation_x=cfg.data.translation_x, translation_y=cfg.data.translation_y, translation_z=cfg.data.translation_z,
                points_3d=cfg.data.points_3d, point_levels_3d=cfg.data.point_levels_3d, point_range_3d=cfg.data.point_range_3d,
                multi_process=cfg.data.multi_process, num_workers=cfg.data.num_workers, all_signals=cfg.data.all_signals,
                use_body_mask = cfg.learning.model.use_body_mask
                )
    # name = f'{cfg.data.translation_x}_{cfg.data.translation_y}_{cfg.data.translation_z}_{cfg.learning.model.prob_dropout}_{cfg.learning.model.signals_dim}_{cfg.learning.model.num_attention_blocks}_{cfg.learning.model.nodes_resistance_network}'

    if cfg.segmentation_model:
        run = wandb.init(project='mask_segmentation',
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        model = hydra.utils.instantiate(cfg.learning.segmentation_model)
        model = seg_training(model, train_dataset, val_dataset, epochs=cfg.learning.training.epochs, batch_size_train=cfg.learning.segmentation_model.batch_size, 
                        lr=cfg.learning.segmentation_model.learning_rate, device=cfg.learning.training.device)
    else:
        run = wandb.init(project='deep_eit',
                     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        model = hydra.utils.instantiate(cfg.learning.model, mask_resolution=cfg.data.mask_resolution)

        if not cfg.inference:
            model = training(model, train_dataset, val_dataset, epochs=cfg.learning.training.epochs, 
                    batch_size_train=cfg.learning.training.batch_size_train, 
                    batch_size_val=cfg.learning.validation.batch_size_val, lr=cfg.learning.training.learning_rate, 
                    loss_lung_multiplier=cfg.learning.training.loss_lung_multiplier, device=cfg.learning.training.device,
                    point_levels_3d=cfg.data.point_levels_3d,
                    downsample_factor_val = cfg.learning.validation.downsample_factor_val,
                    output_dir=output_dir)
            model.load_state_dict(torch.load(os.path.join(output_dir, 'model.pt')))
        else:
            model.load_state_dict(torch.load(os.path.join(cfg.inference_path,'model.pt')))
        testing(model, test_dataset, batch_size=cfg.learning.testing.batch_size_test, device=cfg.learning.training.device, 
                downsample_factor_test=cfg.learning.testing.downsample_factor_test)

if __name__ == '__main__':
    set_seeds(12)
    main()

