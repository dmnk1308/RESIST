import torch
import torch.nn as nn
import numpy as np
from model.model_blocks import *

def positional_encoding(tensor, num_encoding_functions=6, freq=2.):
    """Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    encoding = []
    frequency_bands = torch.pi * (freq ** torch.linspace(
        -1.0,
        num_encoding_functions - 2,
        num_encoding_functions,
        dtype=tensor.dtype,
        device=tensor.device,
    ))
    
    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

class AttentionModel(nn.Module):
    def __init__(
        self,
        num_encoding_functions_electrodes=10,
        num_encoding_functions_points=10,
        num_electrodes=16,
        num_attention_blocks=4,
        attention_dim=128,
        num_linear_output_blocks=4,
        linear_output_channels=512,
        prob_dropout=1.0,
        emb_dropout=0.2,
        training=True,
        use_tissue_embedding=False,
        num_tissue_classes=6,
        use_body_only=True,
        dropout_attention=0.1,
        signal_emb=4,
        pos_encoding=True,
        attention_on="signal",
        **kwargs
    ):
        '''
        RESIST model
        '''
        super(AttentionModel, self).__init__()
        self.num_encoding_functions_electrodes = num_encoding_functions_electrodes
        self.num_encoding_functions_points = num_encoding_functions_points
        self.attention_on = attention_on
        self.num_electrodes = num_electrodes
        self.num_pe_electrodes = 4
        self.pos_encoding = pos_encoding

        # input dimension for linear processing before self attention
        if attention_on == "sequence":
            self.dim_electrodes_pe = (
                13
                * self.num_pe_electrodes
                * 3
                * (2 * num_encoding_functions_electrodes)
            )
        elif attention_on == "signal":
            self.dim_electrodes_pe = (
                self.num_pe_electrodes * 3 * (2 * num_encoding_functions_electrodes)
            )
        else:
            Exception('attention_on must be either "sequence" or "signal"')
        # input dimension for linear processing of embeddings
        self.dim_signal_emb_out = (
            self.num_pe_electrodes * 3 * (2 * num_encoding_functions_electrodes)
        )

        self.attention_dim = attention_dim
        self.prob_dropout = float(prob_dropout)
        self.num_linear_output_blocks = num_linear_output_blocks
        self.training = training
        self.use_tissue_embedding = use_tissue_embedding
        self.use_body_only = use_body_only
        self.emb_dropout = emb_dropout

        # SIGNAL EMBEDDING
        self.signal_embeddings = nn.Embedding(int(4 * 16 * 13), signal_emb)
        self.signal_linear_embedding = nn.Linear(signal_emb, self.dim_signal_emb_out)

        # PREPROCESSING
        self.linear_signals = nn.Linear(self.dim_electrodes_pe, attention_dim)
        self.linear_signals_ln = nn.LayerNorm(attention_dim)

        # SELF ATTENTION
        self.self_attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.self_attention.append(
                SelfAttentionBlock(
                    dim_kqv=attention_dim, heads=8, dropout=dropout_attention
                )
            )
        self.self_attention = nn.Sequential(*self.self_attention)

        # ATTENTION
        points_dim = 3 * 2 * num_encoding_functions_points
        self.linear_points = nn.Linear(points_dim, attention_dim)
        self.linear_points_ln = nn.LayerNorm(attention_dim)
        self.attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention.append(
                AttentionBlock(
                    dim_q=attention_dim,
                    dim_kv=attention_dim,
                    heads=8,
                    dropout=dropout_attention,
                )
            )

        # TISSUE EMBEDDING
        if use_tissue_embedding:
            if use_body_only:
                num_tissue_classes = 2
            self.tissue_embedding = nn.Embedding(num_tissue_classes, points_dim)
            self.point_self_attention = SelfAttentionBlock(
                in_dim=points_dim,
                dim_kqv=attention_dim,
                heads=8,
                dropout=dropout_attention,
            )

        # LINEAR PROCESSING
        self.linear_out = nn.ModuleList()
        for i in range(num_linear_output_blocks):
            if i == 0:
                self.linear_out.append(nn.Linear(attention_dim, linear_output_channels))
            elif i == num_linear_output_blocks - 1:
                self.linear_out.append(nn.Linear(linear_output_channels, 1))
            else:
                self.linear_out.append(
                    nn.Linear(linear_output_channels, linear_output_channels)
                )

    def forward(
        self,
        signals,
        electrodes,
        xy,
        tissue=None,
        return_attention_weights=False,
        training=False,
        **kwargs
    ):
        # signals:      bx4x16x13
        # electrodes:   bx4x16x13x4x3
        # x:            bxnx3
        # targets:      bxnx1

        b = xy.shape[0]
        n = xy.shape[1]
        num_electrodes = electrodes.shape[2] * electrodes.shape[1]
        signals = signals.reshape(b, -1, 16, 13)
        # SIGNAL EMBEDDING
        signals_emb = (
            torch.arange(int(signals.shape[1] * signals.shape[2] * signals.shape[3]), device=signals.device)
            .reshape(1, num_electrodes, 13)
            .repeat(b, 1, 1)
            .type(torch.int64)
        )

        electrodes = electrodes.reshape(
            b, num_electrodes, 13, self.num_pe_electrodes, electrodes.shape[-1]
        ).float()
        signals = signals.reshape(b, num_electrodes, 13).float()

        signals_emb = self.signal_embeddings(signals_emb)

        if training and self.prob_dropout > 0.0:
            check_not_zero = False
            while not check_not_zero:
                dropout_electrodes = torch.bernoulli(
                    torch.full((num_electrodes,), 1 - self.prob_dropout)
                ).bool()
                num_electrodes = torch.sum(dropout_electrodes)
                check_not_zero = num_electrodes > 0
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
            electrodes = electrodes[dropout_electrodes].reshape(
                b, num_electrodes, 13, self.num_pe_electrodes, electrodes.shape[-1]
            )
            signals = signals[dropout_electrodes].reshape(b, num_electrodes, 13)
            signals_emb = signals_emb[dropout_electrodes].reshape(
                b, num_electrodes, 13, signals_emb.shape[-1]
            )
        signals = signals.unsqueeze(-1) + signals_emb
        signals = signals.reshape(b, num_electrodes, 13, -1)
        signals = self.signal_linear_embedding(signals)

        if self.pos_encoding:
            electrodes = positional_encoding(
                electrodes,
                num_encoding_functions=self.num_encoding_functions_electrodes,
            ).reshape(b, num_electrodes, 13, -1)
            signals = electrodes + signals  # .unsqueeze(-1)
        if self.attention_on == "sequence":
            signals = signals.reshape(b, num_electrodes, -1)
        elif self.attention_on == "signal":
            signals = signals.reshape(b, int(num_electrodes * 13), -1)
        signals = self.linear_signals(signals)
        # signals = self.linear_signals_ln(signals)
        signals = self.self_attention(signals)

        # point-signals attention
        xy = positional_encoding(
            xy, num_encoding_functions=self.num_encoding_functions_points
        )
        xy = xy.reshape(b, n, xy.shape[-1])  # bxnx(3x2xnum_encoding_functions)
        xy = self.linear_points(xy)
        # xy = self.linear_points_ln(xy)
        if self.use_tissue_embedding:
            xy_emb = torch.nn.functional.dropout(
                self.tissue_embedding(tissue).reshape(xy.shape) * 0.1,
                p=self.emb_dropout,
                training=self.training,
            )
            xy = xy + xy_emb
            xy = self.point_self_attention(xy)
        attention_weights = []
        for attention in self.attention:
            xy, w = attention(signals, xy, return_weights=return_attention_weights)
            if return_attention_weights:
                attention_weights.append(w.detach().cpu())
        for i, linear_out in enumerate(self.linear_out):
            xy = linear_out(xy)
            xy = nn.functional.relu(xy)  # conductivity is > 0
        xy = xy.reshape(xy.shape[0], -1, 1)
        if return_attention_weights:
            return xy, attention_weights
        return xy
