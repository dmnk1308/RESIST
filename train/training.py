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

def training(model, train_dataset, val_dataset, epochs, batch_size_train, lr, device, batch_size_val=2):
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
        epoch_loss_train, model, optimizer = training_step(model, train_dataloader, optimizer, lr, loss, device, epoch)
        epoch_loss_val, epoch_lung_loss_val = validation_step(model, val_dataloader, loss, device, cmap)
        pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {epoch_loss_train:.5f}, Val Loss: {epoch_loss_val:.5f}, Val Lung Loss: {epoch_lung_loss_val:.5f}")#, Val Masked MSE: {masked_loss_val:.5f}")
        wandb.log({"train_loss": epoch_loss_train, "val_loss": epoch_loss_val,"val_lung_loss": epoch_lung_loss_val})#, "masked_mse_val": masked_loss_val})
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            torch.save(model.state_dict(), 'model.pt')
    return model

def training_step(model, dataloader, optimizer, lr, loss, device, epoch):
    model.train()
    model.to(device)
    epoch_loss = 0
    for i, (points, weights, signals, electrodes, mask , targets) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     weights=weights.to(device))
        l = loss(pred, targets.to(device))
        lung_points = (targets < 0.2) * (targets > 0.05)
        l[lung_points.squeeze()] *= 2
        l = l.mean()
        l.backward()
        optimizer.step()
        epoch_loss += l.detach().cpu()
    epoch_loss /= (i+1)
    return epoch_loss, model, optimizer

def validation_step(model, dataloader, loss, device, cmap):
    model.eval()
    model.to(device)
    epoch_loss = 0
    lung_loss = 0
    # masked_loss = 0
    preds = []
    targets = []
    num_val = 0
    plotted = False
    for i, (points, weights, signals, electrodes, mask, target) in enumerate(dataloader):
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     weights=weights.to(device))
        lung_points = (target < 0.2) * (target > 0.05)
        epoch_loss += loss(pred.detach().cpu(), target).mean()
        if torch.sum(lung_points) > 0:
            lung_loss += loss(pred.detach().cpu()[lung_points], target[lung_points]).mean()
        # m_l = masked_mse(pred.reshape(-1, 512, 512, 1).detach().cpu(), target.reshape(-1, 512, 512, 1), mask.int().reshape(-1, 512, 512, 1))
        if num_val < 4:
            preds.append(pred.detach().cpu().reshape(-1, 512, 512, 1))
            targets.append(target.detach().cpu().reshape(-1, 512, 512, 1))
            num_val += preds[-1].shape[0]
            # log qualitative results
        if num_val >= 4 and not plotted:
            preds = torch.cat(preds[:4])
            targets = torch.cat(targets[:4])
            log_heatmaps(targets, preds, n_examples=1, n_res=4, cmap=cmap)
            plotted = True
        # masked_loss += m_l
    epoch_loss /= (i+1)
    lung_loss /= (i+1)
    # masked_loss /= (i+1)
    return epoch_loss, lung_loss#, masked_loss
