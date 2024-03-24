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
from utils.helper import log_heatmaps

def testing(model, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss = nn.MSELoss()
    model.eval()
    model.to(device)
    preds = []
    targets = []
    for i, (points, weights, signals, electrodes, mask, target) in enumerate(dataloader):
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     weights=weights.to(device))
        preds.append(pred.detach().cpu())
        targets.append(target.detach().cpu())
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    test_loss = loss(preds.to(device), targets.to(device))
    # preds_closest = closest_class(preds)
    preds = preds.reshape(-1, 512, 512, 1)
    targets = targets.reshape(-1, 512, 512, 1)
    print('Test Loss: ', test_loss)

    # log qualitative results
    log_heatmaps(targets, preds)
    wandb.log({"test_loss": test_loss})
