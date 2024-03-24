import torch
import torch.nn as nn
from model.positional_encoding import positional_encoding

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

class Model(nn.Module):
    def __init__(self, conv_out_channels=32, conv_in_channels=1, conv_blocks=5, conv_features=512, 
                 nodes_resistance_network=512, resistance_network_blocks=4, 
                 nodes_signal_network=512, signal_network_blocks=4,
                 mask_resolution=512, num_encoding_functions=6, num_electrodes=16, no_weights=False, **kwargs):
        super(Model, self).__init__()
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