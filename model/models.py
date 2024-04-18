import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.positional_encoding import positional_encoding
from model.model_blocks import *

class Model(nn.Module):
    def __init__(self, conv_out_channels=32, conv_in_channels=1, conv_blocks=5, conv_features=512, 
                 nodes_resistance_network=512, resistance_network_blocks=4, 
                 nodes_signal_network=512, signal_network_blocks=4,
                 mask_resolution=512, num_encoding_functions=6, num_electrodes=16, no_weights=False, **kwargs):
        super(Model, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.no_weights = no_weights
        # process mask
        self.mask_conv = nn.ModuleList()
        for i in range(conv_blocks):
            self.mask_conv.append(ResBlock(conv_in_channels, conv_out_channels))
            conv_in_channels = conv_out_channels
            conv_out_channels = conv_out_channels*2
            if i < conv_blocks-1:
                self.mask_conv.append(nn.MaxPool2d(2))
        self.mask_conv = nn.Sequential(*self.mask_conv)
        mask_feature_resolution = int(mask_resolution/(2**(conv_blocks-1)))
        self.mask_linear = nn.Linear(conv_in_channels*(mask_feature_resolution**2), conv_features) # flatten conv output

        # process encoded coordinates
        resistance_network_input_dim = conv_features + (2 * (2*num_encoding_functions)) + (num_electrodes * 3 * (2*num_encoding_functions))
        self.resistance_network = nn.ModuleList()
        for i in range(resistance_network_blocks):
            self.resistance_network.append(nn.Linear(resistance_network_input_dim, nodes_resistance_network))
            self.resistance_network.append(nn.ReLU())
            resistance_network_input_dim = nodes_resistance_network
        self.resistance_network = nn.Sequential(*self.resistance_network)

        # process signals
        signal_network_input_dim = nodes_resistance_network + (num_electrodes**2) + conv_features # 16*16=signals
        self.signal_network = nn.ModuleList()
        for i in range(signal_network_blocks):
            self.signal_network.append(nn.Linear(signal_network_input_dim, nodes_signal_network))
            self.signal_network.append(nn.ReLU())
            signal_network_input_dim = nodes_signal_network
        self.signal_network.append(nn.Linear(nodes_signal_network, 1))
        self.signal_network = nn.Sequential(*self.signal_network)

    def forward(self, signals, masks, electrodes, xy, weights, **kwargs):

        expand_dim = xy.shape[1]
        
        # process mask
        masks = masks.reshape(masks.shape[0], 1, masks.shape[1], masks.shape[2])
        masks = self.mask_conv(masks)
        masks = masks.view(masks.shape[0], -1) # flatten
        masks = self.mask_linear(masks)
        masks = masks.unsqueeze(1).expand(masks.shape[0], expand_dim, masks.shape[1])

        # process encoded coordinates
        electrodes = positional_encoding(electrodes, num_encoding_functions=self.num_encoding_functions)
        electrodes = electrodes.reshape(electrodes.shape[0], -1)
        electrodes = electrodes.unsqueeze(1).expand(electrodes.shape[0], expand_dim, electrodes.shape[-1])
        # electrodes = electrodes.reshape(-1, electrodes.shape[-1])

        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(xy.shape[0], -1, xy.shape[-1])

        # masks = masks.reshape(xy.shape[0], -1)

        x = torch.cat((xy, electrodes, masks), dim=2)
        x = self.resistance_network(x)

        # process signals
        signals = signals.reshape(-1, 1, 16*16)
        signals = signals.expand(-1, expand_dim, -1)
        if not self.no_weights:
            weights = weights.reshape(-1, expand_dim, 16*16)
            signals = signals * weights

        x = torch.cat((x, signals, masks), dim=2)
        x = self.signal_network(x)
        
        return x
    
class BaseModel(nn.Module):
    def __init__(self,  
                 nodes_resistance_network=512, resistance_network_blocks=4, 
                 nodes_signal_network=512, signal_network_blocks=4,
                 num_encoding_functions=6, num_electrodes=16, **kwargs):
        super(BaseModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        # process signals and electrode positions naivly together
        self.signal_network = nn.ModuleList()
        dim_input_signal_network = 16 * 16 + (num_electrodes * 3 * (2*num_encoding_functions))
        for i in range(signal_network_blocks):
            self.signal_network.append(nn.Linear(dim_input_signal_network, nodes_signal_network))
            dim_input_signal_network = nodes_signal_network
            if i==signal_network_blocks-1:
                break
            self.signal_network.append(nn.ReLU())
        self.signal_network = nn.Sequential(*self.signal_network)

        # process point coordinates and features
        self.resistance_network = nn.ModuleList()
        dim_input_resistance_network = nodes_signal_network + (2 * (2*num_encoding_functions))
        for i in range(resistance_network_blocks):
            self.resistance_network.append(nn.Linear(dim_input_resistance_network, nodes_resistance_network))
            dim_input_resistance_network = nodes_resistance_network
            self.resistance_network.append(nn.ReLU())
        self.resistance_network.append(nn.Linear(nodes_resistance_network, 1))
        self.resistance_network = nn.Sequential(*self.resistance_network)

    def forward(self, signals, electrodes, xy, **kwargs):
        expand_dim = xy.shape[1]
        # process electrodes and signals
        electrodes = positional_encoding(electrodes, num_encoding_functions=self.num_encoding_functions)
        electrodes = electrodes.reshape(electrodes.shape[0], -1)
        electrodes = electrodes.unsqueeze(1).expand(electrodes.shape[0], expand_dim, electrodes.shape[-1])
        signals = signals.reshape(-1, 1, 16*16)
        signals = signals.expand(-1, expand_dim, -1)
        signals = torch.cat((electrodes, signals), dim=2)
        signals = self.signal_network(signals)
        # combine with query coordinate
        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(xy.shape[0], expand_dim, xy.shape[-1])
        x = torch.cat((xy, signals), dim=2)
        x = self.resistance_network(x)
        return x
    

# ATTENTION MODEL
class AttentionModel(nn.Module):
    def __init__(self, num_encoding_functions=6, num_electrodes=16, nodes_resistance_network=512, 
                 num_attention_blocks= 4, num_linear_output_blocks=4, **kwargs):
        super(AttentionModel, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.num_electrodes = num_electrodes
        self.dim_electrodes_pe = 3 * (2*num_encoding_functions)

        self.linear_signals = nn.Linear(16, self.dim_electrodes_pe)
        self.linear_signals_2 = nn.Linear(self.dim_electrodes_pe, self.dim_electrodes_pe)

        self.self_attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.self_attention.append(SelfAttentionBlock(in_dim=self.dim_electrodes_pe, heads=8))
        self.self_attention = nn.Sequential(*self.self_attention)

        self.attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention.append(AttentionBlock(in_dim=self.dim_electrodes_pe, in_dim_q=2*2*num_encoding_functions, heads=8))

        self.linear_out = nn.ModuleList()
        for i in range(num_linear_output_blocks):
            if i==0:
                self.linear_out.append(nn.Linear(2*2*num_encoding_functions, nodes_resistance_network))
            elif i==num_linear_output_blocks-1:
                self.linear_out.append(nn.Linear(nodes_resistance_network, 1))
            else:
                self.linear_out.append(nn.Linear(nodes_resistance_network, nodes_resistance_network))
        
    def forward(self, signals, electrodes, xy, **kwargs):
        # signals:      bx16x16
        # electrodes:   bx16x3
        # x:            bxnx2

        n = xy.shape[1]

        # electrodes-signals self-attention
        electrodes = positional_encoding(electrodes, num_encoding_functions=self.num_encoding_functions)
        electrodes = electrodes.reshape(electrodes.shape[0], self.num_electrodes, -1) # bx16x(3*2*num_encoding_functions)
        # electrodes = concatenate_electrode_pairs(electrodes) # bx16x(2*3*2*num_encoding_functions)
        signals = self.linear_signals(signals) # bx16x(2*3*2*num_encoding_functions)
        signals = nn.functional.relu(signals)
        signals = self.linear_signals_2(signals) # bx16x(2*3*2*num_encoding_functions)

        # signals = nn.functional.relu(signals)
        # signals = nn.functional.layer_norm(signals, signals.shape[1:])
        signals = electrodes + signals
        signals = self.self_attention(signals)

        # point-signals attention
        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(xy.shape[0], n, xy.shape[-1]) # bxnx(2x2xnum_encoding_functions)
        for attention in self.attention:
            xy = attention(signals, xy)
        for i, linear_out in enumerate(self.linear_out[:-1]):
            xy = linear_out(xy)
            # xy = nn.functional.layer_norm(xy, xy.shape[1:])
            xy = nn.functional.relu(xy)
        xy = self.linear_out[-1](xy)
        return xy
    
class AttentionModelMask(nn.Module):
    def __init__(self, num_encoding_functions=6, num_electrodes=16, nodes_resistance_network=512, mask_resolution=128, conv_blocks=3, conv_in_channels=16, conv_out_channels=32, conv_features=32, **kwargs):
        super(AttentionModelMask, self).__init__()
        self.num_encoding_functions = num_encoding_functions
        self.num_electrodes = num_electrodes
        self.dim_electrodes_pe = 3 * (2*num_encoding_functions)

        # Mask convolution
        self.mask_conv = nn.ModuleList()
        for i in range(conv_blocks):
            self.mask_conv.append(ResBlock(conv_in_channels, conv_out_channels))
            conv_in_channels = conv_out_channels
            conv_out_channels = conv_out_channels*2
            if i < conv_blocks-1:
                self.mask_conv.append(nn.MaxPool2d(2))
        self.mask_conv = nn.Sequential(*self.mask_conv)
        mask_feature_resolution = int(mask_resolution/(2**(conv_blocks-1)))
        self.mask_linear = nn.Linear(conv_in_channels*(mask_feature_resolution**2), conv_features)        


        self.linear_signals = nn.Linear(16, 2 * self.dim_electrodes_pe)

        self.self_attention_1 = SelfAttentionBlock(in_dim=2*self.dim_electrodes_pe, heads=8)
        self.self_attention_2 = SelfAttentionBlock(in_dim=2*self.dim_electrodes_pe, heads=8)
        self.self_attention_3 = SelfAttentionBlock(in_dim=2*self.dim_electrodes_pe, heads=8)

        self.attention_1 = AttentionBlock(in_dim=2*self.dim_electrodes_pe, in_dim_q=2*2*num_encoding_functions, heads=8)
        self.attention_2 = AttentionBlock(in_dim=2*self.dim_electrodes_pe, in_dim_q=2*2*num_encoding_functions, heads=8)
        self.attention_3 = AttentionBlock(in_dim=2*self.dim_electrodes_pe, in_dim_q=2*2*num_encoding_functions, heads=8)

        self.linear_sequential = nn.Sequential(
            nn.Linear(2*2*num_encoding_functions + conv_features, nodes_resistance_network),
            nn.ReLU(),
            nn.Linear(nodes_resistance_network, nodes_resistance_network),
            nn.ReLU(),
            nn.Linear(nodes_resistance_network, 1),
        )
        
    def forward(self, signals, electrodes, masks, xy, **kwargs):
        # signals:      bx16x16
        # electrodes:   bx16x3
        # masks:        bx128x128
        # x:            bxnx2

        n = xy.shape[1]

        # process mask
        masks = masks.reshape(masks.shape[0], 1, masks.shape[1], masks.shape[2])
        masks = self.mask_conv(masks)
        masks = masks.view(masks.shape[0], -1) # flatten
        masks = self.mask_linear(masks)
        masks = masks.unsqueeze(1).expand(masks.shape[0], n, masks.shape[1])

        # electrodes-signals self-attention
        electrodes = positional_encoding(electrodes, num_encoding_functions=self.num_encoding_functions)
        electrodes = electrodes.reshape(electrodes.shape[0], self.num_electrodes, -1) # bx16x(3*2*num_encoding_functions)
        electrodes = concatenate_electrode_pairs(electrodes) # bx16x(6*2*num_encoding_functions)
        signals = self.linear_signals(signals) # bx16x(6*2*num_encoding_functions)
        signals = electrodes + signals
        signals = self.self_attention_1(signals)
        signals = self.self_attention_2(signals)
        signals = self.self_attention_3(signals)

        # point-signals attention
        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(xy.shape[0], n, xy.shape[-1]) # bxnx(2x2xnum_encoding_functions)
        xy = self.attention_1(signals, xy)
        xy = self.attention_2(signals, xy)
        xy = self.attention_3(signals, xy)
        xy = torch.cat((xy, masks), dim=2)
        xy = self.linear_sequential(xy)
        return xy

def concatenate_electrode_pairs(electrodes):
    # concatenate each row with the next row
    electrodes_new = torch.cat((electrodes[:,:-1], electrodes[:,1:]), dim=2)
    last_row = torch.cat((electrodes[:,-1], electrodes[:,0]), dim=1)
    electrodes_new = torch.cat((electrodes_new, last_row.unsqueeze(1)), dim=1)
    return electrodes_new

class AttentionModelEmbedding(nn.Module):
    def __init__(self, num_encoding_functions=6, num_electrodes=16, nodes_resistance_network=512, 
                 num_attention_blocks= 4, electrode_levels=4, num_level_embedding=32, num_position_embedding=32,
                 num_linear_output_blocks=3, **kwargs):
        super(AttentionModelEmbedding, self).__init__()
        self.num_electrodes = num_electrodes
        self.electrode_level_embedding = nn.Embedding(electrode_levels, num_level_embedding)
        self.electrode_position_embedding = nn.Embedding(num_electrodes, num_position_embedding)
        self.num_encoding_functions = num_encoding_functions
        self.linear_signals = nn.Linear(16, num_level_embedding+num_position_embedding)
        self.linear_signals_2 = nn.Linear(num_level_embedding+num_position_embedding, num_level_embedding+num_position_embedding)

        self.self_attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.self_attention.append(SelfAttentionBlock(in_dim=num_level_embedding+num_position_embedding, heads=8))
        self.self_attention = nn.Sequential(*self.self_attention)

        self.attention = nn.ModuleList()
        for _ in range(num_attention_blocks):
            self.attention.append(AttentionBlock(in_dim=num_level_embedding+num_position_embedding, in_dim_q=2*2*num_encoding_functions, heads=8))

        self.linear_out = nn.ModuleList()
        for i in range(num_linear_output_blocks):
            if i==0:
                self.linear_out.append(nn.Linear(2*2*num_encoding_functions, nodes_resistance_network))
            elif i==num_linear_output_blocks-1:
                self.linear_out.append(nn.Linear(nodes_resistance_network, 1))
            else:
                self.linear_out.append(nn.Linear(nodes_resistance_network, nodes_resistance_network))

    def forward(self, signals, electrodes, xy, **kwargs):
        # signals:      bx16x16
        # electrodes:   bx16x3
        # x:            bxnx2

        n = xy.shape[1]
        batch_size = xy.shape[0]

        # electrodes-signals self-attention
        electrodes_emb_xy = self.electrode_position_embedding(torch.arange(self.num_electrodes).int().to(xy.device))
        electrodes_emb_xy = electrodes_emb_xy.unsqueeze(0).expand(batch_size, -1, -1) # bx16xembedding_dim
        electrodes_emb_level = self.electrode_level_embedding(electrodes.int())
        electrodes_emb_level = electrodes_emb_level.unsqueeze(1).expand(-1, self.num_electrodes, -1) # bx16xembedding_dim
        electrodes = torch.cat((electrodes_emb_level, electrodes_emb_xy), dim=-1)
        signals = self.linear_signals(signals) # bx16x(embedding_dim+embedding_dim)
        signals = nn.functional.relu(signals)
        signals = self.linear_signals_2(signals) # bx16x(2*3*2*num_encoding_functions)
                
        signals = electrodes + signals
        signals = self.self_attention(signals)

        # point-signals attention
        xy = positional_encoding(xy, num_encoding_functions=self.num_encoding_functions)
        xy = xy.reshape(xy.shape[0], n, xy.shape[-1]) # bxnx(2x2xnum_encoding_functions)
        for attention in self.attention:
            xy = attention(signals, xy)
        for i, linear_out in enumerate(self.linear_out[:-1]):
            xy = linear_out(xy)
            xy = nn.functional.layer_norm(xy, xy.shape[1:])
            xy = nn.functional.relu(xy)
        xy = self.linear_out[-1](xy)
        return xy