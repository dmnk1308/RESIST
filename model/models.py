import torch
import torch.nn as nn
import numpy as np
from model.model_blocks import *


class Resist(nn.Module):
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
        super(Resist, self).__init__()
        self.num_encoding_functions_electrodes = num_encoding_functions_electrodes
        self.num_encoding_functions_points = num_encoding_functions_points
        self.attention_on = attention_on
        self.num_electrodes = num_electrodes
        self.num_pe_electrodes = 4
        self.pos_encoding = pos_encoding
        self.use_body_only = use_body_only        
        self.encoder = ResistEncoder(
            num_encoding_functions_electrodes=num_encoding_functions_electrodes,
            num_encoding_functions_points=num_encoding_functions_points,
            num_electrodes=num_electrodes,
            num_attention_blocks=num_attention_blocks,
            attention_dim=attention_dim,
            num_linear_output_blocks=num_linear_output_blocks,
            prob_dropout=prob_dropout,
            emb_dropout=emb_dropout,
            use_tissue_embedding=use_tissue_embedding,
            use_body_only=use_body_only,
            dropout_attention=dropout_attention,
            signal_emb=signal_emb,
            pos_encoding=pos_encoding,
            attention_on=attention_on,
            **kwargs
        )
        self.decoder = ResistDecoder(
            num_encoding_functions_points=num_encoding_functions_points,
            num_electrodes=num_electrodes,
            num_attention_blocks=num_attention_blocks,
            attention_dim=attention_dim,
            num_linear_output_blocks=num_linear_output_blocks,
            emb_dropout=emb_dropout,
            use_tissue_embedding=use_tissue_embedding,
            use_body_only=use_body_only,
            dropout_attention=dropout_attention,
            pos_encoding=pos_encoding,
            **kwargs
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

        signals = self.encoder(signals, electrodes, training=training, **kwargs)
        pred = self.decoder(signals, xy, training=training, **kwargs)
        return pred


############### MEAN LUNG COND PRED MODEL ###############
class ResistMeanLung(nn.Module):
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
        super(ResistMeanLung, self).__init__()
        self.num_encoding_functions_electrodes = num_encoding_functions_electrodes
        self.num_encoding_functions_points = num_encoding_functions_points
        self.attention_on = attention_on
        self.num_electrodes = num_electrodes
        self.num_pe_electrodes = 4
        self.pos_encoding = pos_encoding
        self.use_body_only = use_body_only        
        self.encoder = ResistEncoder(
            num_encoding_functions_electrodes=num_encoding_functions_electrodes,
            num_encoding_functions_points=num_encoding_functions_points,
            num_electrodes=num_electrodes,
            num_attention_blocks=num_attention_blocks,
            attention_dim=attention_dim,
            num_linear_output_blocks=num_linear_output_blocks,
            prob_dropout=prob_dropout,
            emb_dropout=emb_dropout,
            use_tissue_embedding=use_tissue_embedding,
            use_body_only=use_body_only,
            dropout_attention=dropout_attention,
            signal_emb=signal_emb,
            pos_encoding=pos_encoding,
            attention_on=attention_on,
            **kwargs
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
        training=False,
        **kwargs
    ):
        # signals:      bx4x16x13
        # electrodes:   bx4x16x13x4x3
        # x:            bxnx3
        # targets:      bxnx1

        signals = self.encoder(signals, electrodes, training=training, **kwargs)
        signals = torch.mean(signals, dim=1)

        # LINEAR OUTPUT
        for i, linear_out in enumerate(self.linear_out):
            signals = linear_out(signals)
            signals = nn.functional.relu(signals)

        return signals


############### LINEAR MODEL ###############
def total_variation_loss(y_pred):
    """
    Calculate the total variation loss which penalizes large differences between adjacent pixels.
    Args:
        y_pred (torch.Tensor): The predicted image (batch_size, 1, height, width)
    Returns:
        torch.Tensor: The total variation loss
    """
    # Shifted images to calculate difference between adjacent pixels
    diff_i = torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :])  # Vertical differences
    diff_j = torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])  # Horizontal differences

    # Total variation loss is the sum of differences
    tv_loss = torch.sum(diff_i) + torch.sum(diff_j)
    return tv_loss

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # Define a simple linear model 
        self.fc = nn.Linear(208*4, 4*512*512)

    def forward(self, x):
        x = self.fc(x)  # (batch_size, 208*4)
        x = x.view(-1, 4, 512, 512)  # (batch_size, 4, 512, 512)
        return x