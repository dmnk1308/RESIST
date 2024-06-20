import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.positional_encoding import positional_encoding
from model.model_blocks import *
from data_processing.helper import combine_electrode_positions

class AttentionModel(nn.Module):
    def __init__(self, num_encoding_functions=6, num_electrodes=16, signals_dim=512, num_attention_blocks=4, 
                 num_attention_nodes=128, num_linear_output_blocks=4, cnn_in_channels=16, cnn_out_channels=32, linear_output_channels=512, 
                 prob_dropout=1., emb_dropout=0.2, training=True, use_cnn=True, use_tissue_embedding=False, num_tissue_classes=6, 
                 use_body_only=True, use_body_mask=False, body_mask_channels=4, body_mask_blocks=5, body_mask_final_channels=6,
                 dropout_attention=0.1,
                 **kwargs):
        super(AttentionModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.num_electrodes = num_electrodes
        self.num_pe_electrodes = 4    
        self.dim_electrodes_pe = 13 * self.num_pe_electrodes * 3 * (2*num_encoding_functions)
        self.signals_dim = signals_dim
        self.prob_dropout = float(prob_dropout)
        self.num_linear_output_blocks = num_linear_output_blocks
        self.training = training
        self.use_cnn = use_cnn
        self.use_tissue_embedding = use_tissue_embedding
        self.use_body_only = use_body_only
        self.emb_dropout = emb_dropout
        self.use_body_mask = use_body_mask
        if not use_cnn:
            cnn_in_channels = 1

        # PREPROCESSING
        self.linear_signals = nn.Linear(self.dim_electrodes_pe, signals_dim)

        # SELF ATTENTION
        self.self_attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.self_attention.append(SelfAttentionBlock(dim_kqv=signals_dim, heads=8, dropout=dropout_attention))
        self.self_attention = nn.Sequential(*self.self_attention)

        # ATTENTION
        points_dim = 3*2*num_encoding_functions
        self.linear_points = nn.Linear(points_dim, signals_dim)
        self.attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention.append(AttentionBlock(dim_q=signals_dim, dim_kv=signals_dim, heads=8, dropout=dropout_attention))

        # TISSUE EMBEDDING
        if use_tissue_embedding:
            if use_body_only:
                num_tissue_classes = 2
            self.tissue_embedding = nn.Embedding(num_tissue_classes, points_dim)
            self.point_self_attention = SelfAttentionBlock(in_dim=points_dim, dim_kqv=num_attention_nodes, heads=8, dropout=dropout_attention)

        # BODY MASK PROCESSING
        if use_body_mask:
            self.body_cnn = Conv3DNet(channels=body_mask_channels, final_out_channels=body_mask_final_channels, 
                                      blocks=body_mask_blocks, downscaled_channels=signals_dim)

        # LINEAR PROCESSING
        self.linear_out = nn.ModuleList()
        for i in range(num_linear_output_blocks):
            if i==0:
                self.linear_out.append(nn.Linear(signals_dim, linear_output_channels))
            elif i==num_linear_output_blocks-1:
                self.linear_out.append(nn.Linear(linear_output_channels, cnn_in_channels))
            else:
                self.linear_out.append(nn.Linear(linear_output_channels, linear_output_channels))

        # CNN
        if use_cnn:
            self.cnn = UNetBLock(in_channels=cnn_in_channels, out_channels=cnn_out_channels)
        
    def forward(self, signals, electrodes, xy, masks=None, tissue=None, **kwargs):
        # signals:      bx4x16x13
        # electrodes:   bx4x16x13x4x3
        # x:            bxnx3
        # targets:      bxnx1
        b = xy.shape[0]
        n = xy.shape[1]

        num_electrodes = electrodes.shape[2]*electrodes.shape[1]
        electrodes = electrodes.reshape(b, num_electrodes, 13, self.num_pe_electrodes, 3).float()
        signals = signals.reshape(b, num_electrodes, 13).float()
        if self.training:
            check_not_zero = False
            while not check_not_zero:
                dropout_electrodes = torch.bernoulli(torch.full((num_electrodes,), 1-self.prob_dropout)).bool()
                num_electrodes = torch.sum(dropout_electrodes)
                check_not_zero = num_electrodes>0
            # Number of times to shuffle
            num_shuffles = electrodes.shape[0]
            # List to store shuffled tensors
            shuffled_tensors = []
            # Loop over the number of shuffles
            for _ in range(num_shuffles):
                # Generate random permutation of indices
                perm_indices = torch.randperm(dropout_electrodes.size(0))
                # Shuffle the tensor using the random permutation
                shuffled_tensor = dropout_electrodes[perm_indices]
                # Append the shuffled tensor to the list
                shuffled_tensors.append(shuffled_tensor)
            # Concatenate the permuted tensors along the specified dimension (0 for concatenating along rows)
            dropout_electrodes = torch.stack(shuffled_tensors, dim=0)
            electrodes = electrodes[dropout_electrodes].reshape(b, num_electrodes, 13, self.num_pe_electrodes, 3)
            signals = signals[dropout_electrodes].reshape(b, num_electrodes, 13)
        electrodes = positional_encoding(electrodes, num_encoding_functions=self.num_encoding_functions).reshape(b, num_electrodes, 13, -1)

        signals = electrodes + signals.unsqueeze(-1)
        signals = self.linear_signals(signals.reshape(b,num_electrodes,-1))
        if self.use_body_mask:
            mask_feat = self.body_cnn(masks.unsqueeze(1))
            signals = signals + mask_feat.unsqueeze(1).repeat(1, num_electrodes, 1)
        signals = self.self_attention(signals)

        # point-signals attention
        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(b, n, xy.shape[-1]) # bxnx(3x2xnum_encoding_functions)
        xy = self.linear_points(xy)
        if self.use_tissue_embedding:
            xy_emb = torch.nn.functional.dropout(self.tissue_embedding(tissue).reshape(xy.shape) * 0.1, p=self.emb_dropout, training=self.training)
            xy = xy + xy_emb
            xy = self.point_self_attention(xy)
        for attention in self.attention:
            xy = attention(signals, xy)
        for i, linear_out in enumerate(self.linear_out):
            xy = linear_out(xy)
            # xy = nn.functional.layer_norm(xy, xy.shape[1:])
            if i != self.num_linear_output_blocks-1:
                xy = nn.functional.relu(xy)
        
        if self.use_cnn:
            xy = xy.reshape(xy.shape[0], int(np.sqrt(n)), int(np.sqrt(n)), -1).moveaxis(-1, 1)
            xy = self.cnn(xy)
        xy = xy.reshape(xy.shape[0], -1, 1)
        return xy
    