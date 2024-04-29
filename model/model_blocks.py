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
    
class SelfAttentionLayer(nn.Module):
    def __init__(self, in_dim, dim_kqv = 128, heads=8):
        super(SelfAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.heads = heads        
        
        # Define query, key, and value linear transformations for each head
        self.query = nn.Linear(in_dim, dim_kqv, bias=False)
        self.key = nn.Linear(in_dim, dim_kqv, bias=False)
        self.value = nn.Linear(in_dim, dim_kqv, bias=False)
        
        self.multi_head = nn.MultiheadAttention(dim_kqv, num_heads=heads, dropout=0.1, batch_first=True)

        self.fc_out = nn.Linear(dim_kqv, in_dim)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        x = self.multi_head(query, key, value, need_weights=False)[0]
        out = self.fc_out(x)
        return out
    
class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 heads=8):
        super(SelfAttentionBlock, self).__init__()
        self.multihead_selfattention = SelfAttentionLayer(in_dim=in_dim, heads=heads)
        self.linear = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        y = self.multihead_selfattention(x) + x
        y = nn.functional.layer_norm(y, y.shape[1:])
        y = self.linear(y) + y
        y = nn.functional.layer_norm(y, y.shape[1:])
        return y

# ATTENTION MODEL
class AttentionLayer(nn.Module):
    def __init__(self, in_dim, in_dim_q, dim_kqv = 128, heads=8):
        super(AttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.heads = heads
        
        # Define query, key, and value linear transformations for each head
        self.query = nn.Linear(in_dim_q, dim_kqv, bias=False)
        self.key = nn.Linear(in_dim, dim_kqv, bias=False)
        self.value = nn.Linear(in_dim, dim_kqv, bias=False)
        
        self.multi_head = nn.MultiheadAttention(dim_kqv, num_heads=heads, dropout=0.1, batch_first=True)

        self.fc_out = nn.Linear(dim_kqv, in_dim_q)
    
    def forward(self, x, y):
        query = self.query(y)
        key = self.key(x)
        value = self.value(x)
        x = self.multi_head(query, key, value, need_weights=False)[0]
        out = self.fc_out(x)
        return out
    
class AttentionBlock(nn.Module):
    def __init__(self, in_dim, in_dim_q, heads=8):
        super(AttentionBlock, self).__init__()
        self.multihead_attention = AttentionLayer(in_dim, in_dim_q=in_dim_q, heads=heads)
        self.linear = nn.Linear(in_dim_q, in_dim_q)

    def forward(self, x, y):
        y = self.multihead_attention(x, y) + y
        y = nn.functional.layer_norm(y, y.shape[1:])
        y = self.linear(y) + y
        y = nn.functional.layer_norm(y, y.shape[1:])
        return y

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