import torch.nn as nn
import torch
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from utils.helper import log_heatmaps, make_cmap

def training(model, train_dataset, val_dataset, epochs, batch_size_train, lr, device, batch_size_val=2):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    print('Done.')
    optimizer = Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(epochs))
    best_val_loss = 100
    cmap = make_cmap()
    for epoch in pbar:
        epoch_loss_train, model, optimizer = training_step(model, train_dataloader, optimizer, lr, device, epoch)
        epoch_loss_val = validation_step(model, val_dataloader, device, cmap)
        pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {epoch_loss_train:.5f}, Val Loss: {epoch_loss_val:.5f}")#, Val Masked MSE: {masked_loss_val:.5f}")
        wandb.log({"train_loss": epoch_loss_train, "val_loss": epoch_loss_val})#, "masked_mse_val": masked_loss_val})
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            torch.save(model.state_dict(), 'seg_model.pt')
    return model

def training_step(model, dataloader, optimizer, lr, device, epoch):
    model.train()
    model.to(device)
    epoch_loss = 0
    for i, (points, weights, signals, electrodes, mask , targets) in enumerate(dataloader):
        optimizer.zero_grad()
        targets = targets.reshape(-1, 1, 512, 512)
        gt_mask = (targets <= 0.2) * (targets >= 0.05)
        targets[gt_mask] = torch.rand_like(targets[gt_mask])*0.25
        pred = model(x=targets.to(device))
        l = iou_loss(pred.flatten(), gt_mask.flatten().float().to(device))
        l.backward()
        optimizer.step()
        epoch_loss += l.detach().cpu()
    epoch_loss /= (i+1)
    return epoch_loss, model, optimizer    

def validation_step(model, dataloader, device, cmap):
    model.eval()
    model.to(device)
    epoch_loss = 0
    for i, (points, weights, signals, electrodes, mask , targets) in enumerate(dataloader):
        targets = targets.reshape(-1, 1, 512, 512)
        gt_mask = (targets <= 0.2) * (targets >= 0.05)
        targets[gt_mask] = torch.rand_like(targets[gt_mask])*0.25
        pred = model(x=targets.to(device))
        l = iou_loss(torch.round(pred.flatten(), decimals=0), gt_mask.flatten().float().to(device))
        epoch_loss += l.detach().cpu()
        if i < 5:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(targets[0].detach().cpu().squeeze(), cmap=cmap)
            axes[1].imshow(torch.round(pred[0].detach().cpu().squeeze(), decimals=0))
            axes[2].imshow(gt_mask[0].detach().cpu().squeeze())
            wandb.log({'Validation Results': fig})
            plt.close(fig)
    epoch_loss /= (i+1)

    return epoch_loss

def iou_loss(pred, target):
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return 1 - intersection / union