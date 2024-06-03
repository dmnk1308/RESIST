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

def testing(model, data, batch_size, device, wandb_log=True, n_levels=6):
    loss = nn.MSELoss(reduction='none')
    model.eval()
    model.to(device)
    cmap = make_cmap()
    preds = []
    targets = []
    if isinstance(data, Dataset):
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        n_levels = data.point_levels_3d
        for points, weights, signals, electrodes, mask, target in tqdm(dataloader):
            pred = model(signals=signals.to(device), 
                        masks=mask.float().to(device), 
                        electrodes=electrodes.to(device), 
                        xy=points.to(device), 
                        weights=weights.to(device),
                        training=False)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        targets = targets.reshape(-1, 512, 512, 1)
        preds = preds.reshape(-1, 512, 512, 1)
        test_loss = loss(preds.to(device), targets.to(device))
        lung_points = (targets <= 0.2) * (targets >= 0.05)
        lung_points = lung_points * (preds<=0.25)
        if torch.sum(lung_points) > 0:
            test_lung_loss = test_loss[lung_points].mean()
        else:
            test_lung_loss = 0
        test_loss = test_loss.mean()
    else:
        pred = model(signals=data[0].to(device), 
                    electrodes=data[1].to(device), 
                    xy=data[2].to(device), training=False).detach().cpu()
        preds = pred.reshape(-1, 512, 512, 1)
        test_loss = np.nan
        test_lung_loss = np.nan
    if wandb_log:
        print('Test Loss: ', test_loss)
        print('Test Lung Loss: ', test_lung_loss)
        wandb.log({"test_loss": test_loss})    
        wandb.log({"test_lung_loss": test_lung_loss})
        # log qualitative results
        log_heatmaps(targets, preds, n_levels=n_levels, cmap=cmap)
    else:
        print('Test Loss: ', test_loss)
        print('Test Lung Loss: ', test_lung_loss)
        return (targets, preds)

