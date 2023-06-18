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
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

resource_path = 'D:/LZH/code/deep_learning/pytorch/'
pic_dir = './data/0_presented_images_800/'
root_dir = os.path.join(resource_path, pic_dir)
sys.path.append(resource_path)

from pretrained_model import AlexNet


f = h5py.File(os.path.join(root_dir, 'image.h5'), 'r')
data = f['image'][0]
data = torch.from_numpy(data)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(transform(data[0][0][0]))
'''
plt.imshow(data)

plt.show()
'''