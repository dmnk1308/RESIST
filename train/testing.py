import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from train.metrics import closest_class
import wandb
import sys
sys.path.append('../')
from utils.helper import log_heatmaps, make_cmap
from scipy.ndimage import binary_erosion

def testing(model, data, batch_size, device, wandb_log=True, point_levels_3d=6, model_3d=False, point_chunks=8, electrode_level_only=False, return_attention_weights=False):
    model.eval()
    loss = nn.MSELoss(reduction='none')
    model.eval()
    model.to(device)
    cmap = make_cmap()
    preds = []
    targets = []
    attention_weights_list = []
    resolution = 512
    if isinstance(data, Dataset):
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for points, signals, electrodes, mask, target, tissue in tqdm(dataloader):
            if electrode_level_only:
                points = points.reshape(dataloader.batch_size, -1, point_levels_3d, 512, 512, 3)
                target = target.reshape(dataloader.batch_size, -1, point_levels_3d, 512, 512, 1)
                levels = torch.arange(point_levels_3d)        
                electrode_levels = torch.linspace(levels[1],levels[-2],4).numpy().astype(int)
                points = points[:,:,electrode_levels].reshape(dataloader.batch_size, -1, 3)
                target = target[:,:,electrode_levels].reshape(dataloader.batch_size, -1, 1)

            points = points.reshape(dataloader.batch_size, -1, 3)
            points_batched = points.chunk(point_chunks, dim=1)
            target = target.reshape(dataloader.batch_size, -1, 1)
            tissue = tissue.reshape(dataloader.batch_size, -1, 1)
            for points in points_batched:
                if not model_3d:
                    signals = signals.reshape(int(signals.shape[0] * 4), -1, signals.shape[2])
                    points = points.reshape(int(points.shape[0] * 4), -1, points.shape[2])[:,:,:2]
                    electrodes = electrodes.reshape(int(electrodes.shape[0] * 4), -1, electrodes.shape[2], electrodes.shape[3], electrodes.shape[4], electrodes.shape[5])
                    target = target.reshape(int(target.shape[0] * 4), -1, target.shape[2])
                pred = model(signals=signals.to(device), 
                            masks=mask.float().to(device), 
                            electrodes=electrodes.to(device), 
                            xy=points.to(device), 
                            tissue=tissue.to(device),
                            training=False,
                            return_attention_weights=return_attention_weights)
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
        lung_masks = (targets <= 0.2) * (targets >= 0.05)
        lung_masks = lung_masks.squeeze().numpy()
        lung_masks = [binary_erosion(lung_mask, structure=np.ones((40,40))).astype(np.uint8) for lung_mask in lung_masks]
        lung_masks = torch.tensor(lung_masks)

        if torch.sum(lung_masks) > 0:
            test_lung_loss = test_loss[lung_masks].mean()
        else:
            test_lung_loss = 0
        test_loss = test_loss.mean()

    else:
        signals, electrodes, points = data[0], data[1], data[2]
        batch_size = points.shape[0]
        points = points.reshape(batch_size, -1, 3)
        points_batched = points.chunk(8, dim=1)
        for points in points_batched:
            pred = model(signals=signals.to(device), 
                        electrodes=electrodes.to(device), 
                        xy=points.to(device), 
                        training=False,
                        return_attention_weights=return_attention_weights)
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
    if wandb_log:
        print('Test Loss: ', test_loss)
        print('Test Lung Loss: ', test_lung_loss)
        wandb.log({"test_loss": test_loss})    
        wandb.log({"test_lung_loss": test_lung_loss})
        # log qualitative results
        log_heatmaps(targets, preds, n_levels=point_levels_3d, cmap=cmap)
    else:
        print('Test Loss: ', test_loss)
        print('Test Lung Loss: ', test_lung_loss)
        return (targets, preds, attention_weights_list)

