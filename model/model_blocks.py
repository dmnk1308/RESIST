import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.layer(x)
        return nn.functional.relu(y + self.res_conv(x))
    
class FeedForward(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_channels, 4*in_channels)
        self.linear2 = nn.Linear(4*in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = nn.functional.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    

class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 dim_kqv = 128,
                 heads=8,
                 dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.multi_head = nn.MultiheadAttention(dim_kqv, num_heads=heads, dropout=dropout, batch_first=True)    
        self.linear = FeedForward(dim_kqv, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_kqv)
        self.layer_norm2 = nn.LayerNorm(dim_kqv)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.multi_head(x, x, x, need_weights=False)[0] + x
        x = self.linear(self.layer_norm2(x)) + x
        return x

class AttentionBlock(nn.Module):
    def __init__(self, 
                 dim_q=128, 
                 dim_kv = 128, 
                 heads=8, 
                 dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.multi_head = nn.MultiheadAttention(dim_q, num_heads=heads, dropout=dropout, batch_first=True, kdim=dim_kv, vdim=dim_kv)    
        self.linear = FeedForward(dim_q, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_q)
        self.layer_norm2 = nn.LayerNorm(dim_q)

    def forward(self, x, y, return_weights=False):
        # x is key, value | y is query
        y = self.layer_norm1(y)
        if return_weights:
            y_tmp, weights = self.multi_head(y, x, x, need_weights=return_weights)
            y = y_tmp + y
        else:
            weights = None
            y = self.multi_head(y, x, x, need_weights=return_weights)[0] + y
        y = self.linear(self.layer_norm2(y)) + y
        return y, weights

class UNetBLock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=3, **kwargs):
        super(UNetBLock, self).__init__()
        self.blocks = blocks
        self.down_layers = nn.ModuleList()
        for i in range(blocks):
            self.down_layers.append(ResNetBlock(in_channels, out_channels))
            in_channels = out_channels
            if i<blocks-1:
                out_channels *= 2
        self.up_layers = nn.ModuleList()
        for i in range(blocks-1):
            out_channels = out_channels // 2
            self.up_layers.append(nn.ModuleList([nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2), ResNetBlock(out_channels*2, out_channels)]))
            if i<blocks-2:
                in_channels = out_channels
        # final out conv
        self.out_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        out_list = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i<self.blocks-1:
                out_list.append(x)
                x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        for i, layer in enumerate(self.up_layers):
            x = layer[0](x)
            x = torch.cat([x, out_list.pop()], dim=1)
            x = layer[1](x)
        x = self.out_conv(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.layer(x)
        return out + self.res_conv(x)

class Conv3DNet(nn.Module):
    def __init__(self, channels, final_out_channels, blocks, downscaled_channels):
        super(Conv3DNet, self).__init__()
        
        self.conv_layer = nn.ModuleList()
        in_channels = 1
        for i in range(blocks):
            self.conv_layer.append(nn.Conv3d(in_channels, channels, kernel_size=3, padding=1))
            self.conv_layer.append(nn.BatchNorm3d(channels))
            self.conv_layer.append(nn.ReLU())
            in_channels = channels
            if i != blocks-1:
                channels = channels*2
                self.conv_layer.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.conv_layer = nn.Sequential(*self.conv_layer)

        self.conv_last = nn.Conv3d(channels, final_out_channels, kernel_size=1)  
        flattened_dim = ((256/(2**(blocks-1)))**3) * final_out_channels
        self.linear = nn.Linear(int(flattened_dim), downscaled_channels)
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.conv_last(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x