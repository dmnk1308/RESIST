import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import sys
sys.path.append('../')
from utils.helper import log_heatmaps, make_cmap
from train.metrics import masked_mse

def training(model, train_dataset, val_dataset, epochs, batch_size_train, lr, device, loss_lung_multiplier=1, batch_size_val=2, point_levels_3d=9, downsample_factor_val=2, output_dir=None):
    print('Initializing Dataloader...', end=' ')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    print('Done.')
    optimizer = Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='none')
    pbar = tqdm(range(epochs))
    best_val_loss = 100
    cmap = make_cmap()
    for epoch in pbar:
        epoch_loss_train, epoch_lung_loss_train, model, optimizer = training_step(model, train_dataloader, optimizer, lr, loss, device, epoch, loss_lung_multiplier)
        torch.cuda.empty_cache()
        epoch_loss_val, epoch_lung_loss_val = validation_step(model, val_dataloader, loss, device, cmap, point_levels_3d=point_levels_3d, downsample_factor_val=downsample_factor_val)
        pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {epoch_loss_train:.5f}, Val Loss: {epoch_loss_val:.5f},  Train Lung Loss: {epoch_lung_loss_train:.5f}, Val Lung Loss: {epoch_lung_loss_val:.5f}")
        wandb.log({"train_loss": epoch_loss_train, "train_lung_loss": epoch_lung_loss_train, "val_loss": epoch_loss_val,"val_lung_loss": epoch_lung_loss_val})
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}
            torch.save(checkpoint, os.path.join(output_dir, 'model.pt'))
    return model

def training_step(model, dataloader, optimizer, lr, loss, device, epoch, loss_lung_multiplier):
    model.train()
    model.to(device)
    epoch_loss = 0
    lung_loss = 0
    for i, (points, signals, electrodes, mask, targets, tissue) in enumerate(dataloader):
        optimizer.zero_grad()
        if model.use_body_only:
            tissue = torch.where(tissue > 0, 1, 0)
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device),
                     tissue=tissue.to(device))
        l = loss(pred, targets.to(device))
        lung_points = (targets <= 0.2) * (targets >= 0.05) #* (pred.detach().cpu()<=0.25)
        l[lung_points.squeeze()] *= loss_lung_multiplier
        l = l.mean()
        l.backward()
        if torch.sum(lung_points) > 0:
            lung_points = (targets <= 0.2) * (targets >= 0.05) * (pred.detach().cpu()<=0.25)
            lung_loss += loss(pred.detach().cpu()[lung_points], targets[lung_points]).mean()
        optimizer.step()
        epoch_loss += l.detach().cpu()
    epoch_loss /= (i+1)
    lung_loss /= (i+1)
    return epoch_loss, lung_loss, model, optimizer

def validation_step(model, dataloader, loss, device, cmap, point_levels_3d=6, n_examples=4, downsample_factor_val=2):
    model.eval()
    model.to(device)
    epoch_loss = 0
    lung_loss = 0
    # masked_loss = 0
    preds = []
    targets = []
    num_val = 0
    n_plotting = int(point_levels_3d*n_examples)
    plotted = False
    down_resolution = 512//downsample_factor_val
    for i, (points, signals, electrodes, mask, target, tissue) in enumerate(dataloader):
        points = points.reshape(dataloader.batch_size, -1, 512, 512, 3)[:,:,::downsample_factor_val,::downsample_factor_val,:].reshape(dataloader.batch_size, -1, 3)
        target = target.reshape(dataloader.batch_size, -1, 512, 512)[:,:,::downsample_factor_val,::downsample_factor_val].reshape(dataloader.batch_size, -1, 1)
        tissue = tissue.reshape(dataloader.batch_size, -1, 512, 512)[:,:,::downsample_factor_val,::downsample_factor_val].reshape(dataloader.batch_size, -1, 1)
        if model.use_body_only:
            tissue = torch.where(tissue > 0, 1, 0)     
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     tissue=tissue.to(device),
                     training=False)
        lung_points = (target <= 0.2) * (target >= 0.05) * (pred.detach().cpu()<=0.25)
        epoch_loss += loss(pred.detach().cpu(), target).mean()
        if torch.sum(lung_points) > 0:
            lung_loss += loss(pred.detach().cpu()[lung_points], target[lung_points]).mean()
        # m_l = masked_mse(pred.reshape(-1, 512, 512, 1).detach().cpu(), target.reshape(-1, 512, 512, 1), mask.int().reshape(-1, 512, 512, 1))
        if num_val < n_plotting:
            preds.append(pred.detach().cpu().reshape(-1, down_resolution, down_resolution, 1))
            targets.append(target.detach().cpu().reshape(-1, down_resolution, down_resolution, 1))
            num_val += preds[-1].shape[0]
            # log qualitative results
        if num_val >= n_plotting and not plotted:
            preds = torch.cat(preds)[:n_plotting]
            targets = torch.cat(targets)[:n_plotting]
            log_heatmaps(targets, preds, n_examples=n_examples, n_levels=point_levels_3d, cmap=cmap, resolution=down_resolution)
            plotted = True

        # masked_loss += m_l
    epoch_loss /= (i+1)
    lung_loss /= (i+1)
    # masked_loss /= (i+1)
    return epoch_loss, lung_loss#, masked_loss
