import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.positional_encoding import positional_encoding
from model.model_blocks import *

class SegmentationModel(nn.Module):
    def __init__(self, in_channel=1, out_channel=16, blocks=5, **kwargs):
        super(SegmentationModel, self).__init__()
        self.unet = UNetBLock(in_channel, out_channel, blocks)
    
    def forward(self, x):
        return nn.functional.sigmoid(self.unet(x))
