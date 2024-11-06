import os
import sys
import torch
sys.path.append("../..")
from data_processing.dataset import load_dataset_3d
from train.training import training
from train.testing import testing
from utils.helper import get_all_cases, set_seeds
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg: DictConfig, inference=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.chdir(script_dir)
    cases = get_all_cases(cfg, base_dir=script_dir, use_raw=True)

    train_dataset, val_dataset, test_dataset = load_dataset_3d(
        cases,
        resolution=cfg.data.resolution,
        base_dir=script_dir,
        raw_data_folder=cfg.data.raw_data_folder,
        processed_data_folder=cfg.data.processed_data_folder,
        dataset_data_folder=cfg.data.dataset_data_folder,
        name_prefix=cfg.data.name_prefix,
        write_dataset=cfg.data.write_dataset,
        write_npz=cfg.data.write_npz,
        overwrite_npz=cfg.data.overwrite_npz,
        n_sample_points=cfg.learning.training.sample_points,
        apply_rotation=cfg.data.apply_rotation,
        apply_subsampling=cfg.data.apply_subsampling,
        apply_translation=cfg.data.apply_translation,
        translation_x=cfg.data.translation_x,
        translation_y=cfg.data.translation_y,
        translation_z=cfg.data.translation_z,
        point_levels_3d=cfg.data.point_levels_3d,
        multi_process=cfg.data.multi_process,
        num_workers=cfg.data.num_workers,
        signal_norm=cfg.data.signal_norm,
        normalize_space = cfg.data.normalize_space,
        level_used=cfg.data.level_used,
        include_resistivities=cfg.data.include_resistivities

    )
    run = wandb.init(
        project="resist",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb,
    )
    set_seeds(cfg.seed)
    model = hydra.utils.instantiate(cfg.learning.model)

    if not cfg.inference:
        model = training(
            model,
            train_dataset,
            val_dataset,
            epochs=cfg.learning.training.epochs,
            batch_size_train=cfg.learning.training.batch_size_train,
            batch_size_val=cfg.learning.validation.batch_size_val,
            lr=cfg.learning.training.learning_rate,
            loss_lung_multiplier=cfg.learning.training.loss_lung_multiplier,
            device=cfg.learning.training.device,
            point_levels_3d=cfg.data.point_levels_3d,
            output_dir=output_dir,
        )
        model.load_state_dict(
            torch.load(os.path.join(output_dir, "model.pt"))["model_state_dict"]
        )
    else:
        model.load_state_dict(
            torch.load(os.path.join(cfg.inference_path, "model.pt"))["model_state_dict"]
        )
    testing(
        model,
        test_dataset,
        batch_size=cfg.learning.testing.batch_size_test,
        device=cfg.learning.training.device
    )


if __name__ == "__main__":
    main()
 