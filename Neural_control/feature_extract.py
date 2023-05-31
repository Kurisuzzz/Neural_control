import sys
import os
import cv2
import numpy as np
import torchvision.models as models
import h5py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


resource_path = 'D:/LZH/code/deep_learning/pytorch/'
sys.path.append(resource_path)

from pretrained_model import AlexNet

f = h5py.File('image.h5', 'r')

alexnet = models.alexnet(pretrained = True)
data = f['image'][:]

f.close()

layers = [4, 7, 9, 11, 12]  # 这些数字表示AlexNet中每个卷积层后的层索引
models = [torch.nn.Sequential(*alexnet.features[:layer+1]) for layer in layers]

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = [model.to(device) for model in models]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for image in data:
    
    image = transform(image).unsqueeze(0).to(device)
    
    for model, feature_dicts in zip(models, feature_dicts):
        features = model(image)
        feature_dicts[image_file] = features.cpu().detach().numpy()

for i, feature_dict in enumerate(feature_dicts):
    torch.save(feature_dict, f'features_conv{i+1}.pth')



