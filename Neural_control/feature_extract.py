import sys
import os
import cv2
import numpy as np
import torchvision.models as models
import h5py
import torch
from tqdm import tqdm
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

resource_path = 'D:/LZH/code/deep_learning/pytorch/'
pic_dir = './data/0_presented_images_800/'
root_dir = os.path.join(resource_path, pic_dir)
sys.path.append(resource_path)

from pretrained_model import AlexNet


f = h5py.File(os.path.join(root_dir, 'image.h5'), 'r')
data = f['image'][:]

alexnet = AlexNet.alexnet()

stim_224 = np.zeros(shape = (data.shape[0], 3, 224, 224))

for i in range(data.shape[0]):
    stim_tf = cv2.resize(data[i], (224, 224))
    stim_224[i] = np.transpose(stim_tf, (2, 0, 1))

table = {
    'conv1': (64, 55, 55),
    'conv2': (192, 27, 27),
    'conv3': (384, 13, 13),
    'conv4': (256, 13, 13),
    'conv5': (256, 13, 13)
}

for i in tqdm(range(1, 6)):
    print(i)
    conv_arr = np.zeros(shape = ((data.shape[0],) + table[f'conv{i}']))
    for j in range(stim_224.shape[0]):
        conv_arr[i] = alexnet(torch.tensor(np.expand_dims(stim_224[j], axis=0)).float(), f'conv{i}').detach().numpy()
    torch.save(conv_arr, os.path.join(root_dir, f"AlexNet_conv{i}_feature.pth"))
'''
alexnet = models.alexnet(pretrained = True)


for name, param in alexnet.named_parameters():
    print(name, param.shape)

print(data.shape)
exit(0)
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


f.close()
'''

