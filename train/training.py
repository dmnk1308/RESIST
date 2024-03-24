import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import sys
sys.path.append('../')
from utils.helper import log_heatmaps

def training(model, train_dataset, val_dataset, epochs, batch_size_train, lr, device, batch_size_val=2):
    print('Initializing Dataloader...', end=' ')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    print('Done.')
    optimizer = Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    pbar = tqdm(range(epochs))
    best_val_loss = 100
    for epoch in pbar:
        epoch_loss_train, model, optimizer = training_step(model, train_dataloader, optimizer, lr, loss, device, epoch)
        epoch_loss_val = validation_step(model, val_dataloader, loss, device)
        pbar.set_description(f"Epoch: {epoch+1}, Train Loss: {epoch_loss_train:.5f}, Val Loss: {epoch_loss_val:.5f}")
        wandb.log({"train_loss": epoch_loss_train, "val_loss": epoch_loss_val})
        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            torch.save(model.state_dict(), 'model.pt')
    return model

def training_step(model, dataloader, optimizer, lr, loss, device, epoch):
    model.train()
    model.to(device)
    epoch_loss = 0
    for i, (points, weights, signals, electrodes, mask, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     weights=weights.to(device))
        l = loss(pred, targets.to(device))
        l.backward()
        optimizer.step()
        epoch_loss += l.detach().cpu()
    epoch_loss /= (i+1)
    return epoch_loss, model, optimizer

def validation_step(model, dataloader, loss, device):
    model.eval()
    model.to(device)
    epoch_loss = 0
    for i, (points, weights, signals, electrodes, mask, targets) in enumerate(dataloader):
        pred = model(signals=signals.to(device), 
                     masks=mask.float().to(device), 
                     electrodes=electrodes.to(device), 
                     xy=points.to(device), 
                     weights=weights.to(device))
        l = loss(pred, targets.to(device))
        if i==0:
            preds = pred.detach().cpu().reshape(-1, 512, 512, 1)
            targets = targets.detach().cpu().reshape(-1, 512, 512, 1)
            # log qualitative results
            log_heatmaps(targets, preds)
        epoch_loss += l.detach().cpu()
    epoch_loss /= (i+1)
    return epoch_loss
