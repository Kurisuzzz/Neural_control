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
from torch.utils.data import Dataset
resource_path = 'D:/LZH/code/deep_learning/pytorch/'
img_path = './data/0_presented_images_800'
sys.path.append(resource_path)

class MyDataset(Dataset):
    def __init__(self, image, response):
        self.image = image
        self.response = response

    def __getitem__(self, item):
        x = self.image[item]
        y = self.response[item]
        return x, y

    def __len__(self):
        return len(self.image)

conv = 'conv1'

img = torch.load(os.path.join(resource_path, img_path, f'AlexNet_{conv}_feature.pth'))

id = h5py.File('C:/Users/admin/Desktop/pytorch_ovo/data/4_L82LL_V4_S03_D250_objects/stimuli/Random_id_80_2021_10_21.mat', 'r')
idx = np.array(id['sampleidlist21']).squeeze().astype('int') - 1

idx, unique_idx = np.unique(idx, return_index=True)
mat_file = h5py.File('C:/Users/admin/Desktop/pytorch_ovo/data/4_L82LL_V4_S03_D250_objects/stimuli/celldataS_80_CalmAn_75_Objects_16_800_80_50_88_62_trial_mean_normal.mat', 'r')
neural_n = np.transpose(np.array(mat_file['celldataS']), (1, 2, 0)).astype('float16')


#dataset_train = MyDataset(data_x_train, data_y_train)
#dataset_val = MyDataset(data_x_val, data_y_val)



'''    
class conv_encoder(nn.Module):
    def __init__(self):
        super(conv_encoder, self).__init__()
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
        
'''