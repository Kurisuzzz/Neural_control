import sys
import os
import cv2
import numpy as np
import torchvision.models as models
import h5py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

resource_path = 'D:/LZH/code/deep_learning/pytorch/'
sys.path.append(resource_path)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def build(dataset): 
        images, response = dataset
        sz = conv_shape
        px_x_conv = int(sz[2])
        px_y_conv = int(sz[1])
        px_conv = px_x_conv * px_y_conv
        conv_flat = torch.reshape(conv_shape, [-1, px_conv, out_channels[-1], 1])
    def forward(self, x):
        W_spatial_flat = torch.reshape(self.W_S, [px_conv, 1, 1, self.num_neurons])
        h_spatial = F.conv2d(conv_flat, W_spatial_flat, stride=1, padding=0)
        
