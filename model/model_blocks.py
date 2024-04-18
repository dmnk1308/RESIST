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