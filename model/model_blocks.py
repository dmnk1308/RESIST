import torch
import torch.nn as nn

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.layer(x)
        return nn.functional.relu(y + self.res_conv(x))


class FeedForward(nn.Module):
    def __init__(self, in_channels, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(in_channels, 4 * in_channels)
        self.linear2 = nn.Linear(4 * in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = nn.functional.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim_kqv=128, heads=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.multi_head = nn.MultiheadAttention(
            dim_kqv, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.linear = FeedForward(dim_kqv, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_kqv)
        self.layer_norm2 = nn.LayerNorm(dim_kqv)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.multi_head(x, x, x, need_weights=False)[0] + x
        x = self.linear(self.layer_norm2(x)) + x
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim_q=128, dim_kv=128, heads=8, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.multi_head = nn.MultiheadAttention(
            dim_q,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
            kdim=dim_kv,
            vdim=dim_kv,
        )
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
            if i < blocks - 1:
                out_channels *= 2
        self.up_layers = nn.ModuleList()
        for i in range(blocks - 1):
            out_channels = out_channels // 2
            self.up_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            in_channels, out_channels, kernel_size=2, stride=2
                        ),
                        ResNetBlock(out_channels * 2, out_channels),
                    ]
                )
            )
            if i < blocks - 2:
                in_channels = out_channels
        # final out conv
        self.out_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        out_list = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i < self.blocks - 1:
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
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding="same"),
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
            self.conv_layer.append(
                nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
            )
            self.conv_layer.append(nn.BatchNorm3d(channels))
            self.conv_layer.append(nn.ReLU())
            in_channels = channels
            if i != blocks - 1:
                channels = channels * 2
                self.conv_layer.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))
        self.conv_layer = nn.Sequential(*self.conv_layer)

        self.conv_last = nn.Conv3d(channels, final_out_channels, kernel_size=1)
        flattened_dim = ((256 / (2 ** (blocks - 1))) ** 3) * final_out_channels
        self.linear = nn.Linear(int(flattened_dim), downscaled_channels)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.conv_last(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x

class ResistEncoder(nn.Module):
    def __init__(
        self,
        num_encoding_functions_electrodes=10,
        num_encoding_functions_points=10,
        num_electrodes=16,
        num_attention_blocks=4,
        attention_dim=128,
        num_linear_output_blocks=4,
        prob_dropout=1.0,
        emb_dropout=0.2,
        use_tissue_embedding=False,
        use_body_only=True,
        dropout_attention=0.1,
        signal_emb=4,
        pos_encoding=True,
        attention_on="signal",
        **kwargs
    ):
        super(ResistEncoder, self).__init__()
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

    def forward(
        self,
        signals,
        electrodes,
        training=False,
        **kwargs
    ):
        # signals:      bx4x16x13
        # electrodes:   bx4x16x13x4x3
        # x:            bxnx3
        # targets:      bxnx1

        b = signals.shape[0]
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
        return signals
    

class ResistDecoder(nn.Module):
    def __init__(
        self,
        num_encoding_functions_points=10,
        num_attention_blocks=4,
        attention_dim=128,
        num_linear_output_blocks=4,
        linear_output_channels=512,
        use_tissue_embedding=False,
        num_tissue_classes=6,
        use_body_only=True,
        emb_dropout=0.2,
        dropout_attention=0.1,
        pos_encoding=True,
        **kwargs
    ):
        '''
        RESIST decoder model
        '''
        super(ResistDecoder, self).__init__()
        self.num_encoding_functions_points = num_encoding_functions_points
        self.use_tissue_embedding = use_tissue_embedding
        self.emb_dropout = emb_dropout
        
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
        xy,
        tissue=None,
        return_attention_weights=False,
        training=False,
        **kwargs
    ):
        b = xy.shape[0]
        n = xy.shape[1]
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
                training=training,
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