import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import sys
import copy

sys.path.append("../")
from utils.helper import log_heatmaps, make_cmap
from data_processing.helper import erode_lung_masks


def testing(
    model,
    data,
    device,
    wandb_log=True,
    point_levels_3d=6,
    point_chunks=8,
    electrode_level_only=False,
    return_attention_weights=False,
    noise=None,
    electrode_noise=None,
    return_loss=False,
    resolution=512
):
    with torch.no_grad():
        model.eval()
        loss = nn.MSELoss(reduction="none")
        model.to(device)
        cmap = make_cmap()
        preds = []
        targets = []
        attention_weights_list = []
        if isinstance(data, Dataset):
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            for i, (points, signals, electrodes, mask, target, tissue) in tqdm(
                enumerate(dataloader), total=len(dataloader)
            ):
                if electrode_level_only:
                    points = points.reshape(
                        dataloader.batch_size, -1, point_levels_3d, 512, 512, 3
                    )
                    target = target.reshape(
                        dataloader.batch_size, -1, point_levels_3d, 512, 512, 1
                    )
                    levels = torch.arange(point_levels_3d)
                    electrode_levels = (
                        torch.linspace(levels[1], levels[-2], 4).numpy().astype(int)
                    )
                    points = points[:, :, electrode_levels].reshape(
                        dataloader.batch_size, -1, 3
                    )
                    target = target[:, :, electrode_levels].reshape(
                        dataloader.batch_size, -1, 1
                    )
                if noise is not None:
                    signals = signals + noise[i].unsqueeze(0).reshape(signals.shape)
                if electrode_noise is not None:
                    electrodes_copy = copy.deepcopy(electrodes).reshape(-1, 3).numpy()
                    unique_electrodes = copy.deepcopy(
                        electrodes[:, :, 0, 0, :].reshape(-1, 3).numpy()
                    )
                    for e in unique_electrodes:
                        idx = np.all(np.equal(e, electrodes_copy).astype(bool), axis=1)
                        electrodes_copy[idx] += np.random.randn(1, 3) * electrode_noise
                    electrodes = torch.from_numpy(electrodes_copy.reshape(electrodes.shape))
                points = points.reshape(dataloader.batch_size, -1, 3)
                points_batched = points.chunk(point_chunks, dim=1)
                target = target.reshape(dataloader.batch_size, -1, 1)
                tissue = tissue.reshape(dataloader.batch_size, -1, 1)
                for points in points_batched:
                    pred = model(
                        signals=signals.to(device),
                        electrodes=electrodes.to(device),
                        xy=points.to(device),
                        tissue=tissue.to(device),
                        training=False,
                        return_attention_weights=return_attention_weights,
                    )
                    if return_attention_weights:
                        pred, attention_weights = pred
                        attention_weights_list.append(attention_weights.detach().cpu())
                    preds.append(pred.detach().cpu())
                targets.append(target)
            preds = torch.cat(preds, dim=1)
            targets = torch.cat(targets)
            targets = targets.reshape(-1, resolution, resolution, 1)
            preds = preds.reshape(-1, resolution, resolution, 1)
            test_loss = loss(preds, targets)
            lung_masks = erode_lung_masks(targets)

            if torch.sum(lung_masks) > 0:
                test_lung_loss = test_loss[lung_masks].mean()
            else:
                test_lung_loss = 0
            test_loss = test_loss.mean()

        else:
            signals, electrodes, points = data[0], data[1], data[2]
            if not isinstance(points, torch.Tensor):
                points = torch.from_numpy(points).float()
            if not isinstance(electrodes, torch.Tensor):
                electrodes = torch.from_numpy(electrodes).float()
            if not isinstance(signals, torch.Tensor):
                signals = torch.from_numpy(signals).float()
            batch_size = signals.shape[0]
            batch_size = points.shape[0]
            points = points.reshape(batch_size, -1, 3)
            points_batched = points.chunk(point_chunks, dim=1)
            for points in points_batched:
                pred = model(
                    signals=signals.to(device),
                    electrodes=electrodes.to(device),
                    xy=points.to(device),
                    training=False,
                    return_attention_weights=False,
                )
                if return_attention_weights:
                    pred, attention_weights = pred
                    attention_weights_list.append(attention_weights)
                pred = pred.detach().cpu()
                preds.append(pred)
            preds = torch.cat(preds, dim=1)
            preds = preds.reshape(-1, resolution, resolution, 1)
            test_loss = np.nan
            test_lung_loss = np.nan
            return (targets, preds, attention_weights_list)
        print(
            "MSE Test Loss: ",
            test_loss,
            "MSE Test Lung Loss: ",
            test_lung_loss,
            "RMSE Test Loss: ",
            np.sqrt(test_loss),
            "RMSE Test Lung Loss: ",
            np.sqrt(test_lung_loss),
        )
        if wandb_log:
            wandb.log({"test_loss": test_loss})
            wandb.log({"test_lung_loss": test_lung_loss})
            # log qualitative results
            log_heatmaps(targets, preds, n_levels=point_levels_3d, cmap=cmap)
        else:
            if return_loss:
                return (test_loss, test_lung_loss)
            else:
                return (targets, preds, attention_weights_list)
