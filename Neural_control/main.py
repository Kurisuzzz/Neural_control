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


resource_path = 'D:/LZH/code/deep_learning/pytorch/'
img_path = './data/0_presented_images_800'
sys.path.append(resource_path)

conv = 'conv1'
img_feature = torch.load(os.path.join(resource_path, img_path, f'AlexNet_{conv}_feature.pth'))

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

neural_matfile = 'data/1_L76LM_V1_S18_D155_objects/celldataS_43_Objects_11_800_80_30_40_trial_mean_normal.mat'
sequence_matfile = 'data/4_L82LL_V4_S03_D250_objects/stimuli/Random_id_80_2021_10_21.mat'


id = h5py.File(os.path.join(resource_path, sequence_matfile), 'r')
idx = np.array(id['sampleidlist21']).squeeze().astype('int') - 1
idx, unique_idx = np.unique(idx, return_index=True)

img = np.array(img_feature)


mat_file = h5py.File(os.path.join(resource_path, neural_matfile), 'r')
neural_n = np.transpose(np.array(mat_file['celldataS']), (2, 1, 0)).astype('float16')
#print(neural_n.shape) (950, 11, 43)

ns_train = np.delete(neural_n[:800], idx, 0).mean(1)
ns_val = np.concatenate((neural_n[:800][idx], neural_n[800:880][unique_idx]), 1)

#print(idx, unique_idx)
ft_train = np.delete(img[:800], idx, 0)
ft_val = img[:800][idx]

print(ft_train.shape, ft_val.shape) #(725, 64, 55, 55), (75, 64, 55, 55)

dataset_train = MyDataset(ft_train, ns_train)
dataset_val = MyDataset(ft_val, ns_val)

device = 'cuda:0'

batch = 16
AlexNet_dict = {
    'conv1': (64, 55, 55),
    'conv2': (192, 27, 27),
    'conv3': (384, 13, 13),
    'conv4': (256, 13, 13),
    'conv5': (256, 13, 13)
}
c = (batch, ) + AlexNet_dict[conv]
num_neurons = neural_n.shape[2]
channels = neural_n.shape[0]
sz = c.get_shape()
px_x_conv = int(sz[2])
px_y_conv = int(sz[1])
px_conv = px_x_conv * px_y_conv


class conv_encoder(nn.Module):
    def __init__(self, c, response):
        super(conv_encoder, self).__init__()
        self.W_spatial = nn.Parameter(torch.Tensor(px_conv, num_neurons))
        torch.nn.init.trunc_normal_(self.W_spatial, mean=0.0, std=0.001)
        
        self.W_features = nn.Parameter(torch.Tensor(channels, num_neurons))
        torch.nn.init.trunc_normal_(self.W_features, mean=0.0, std=0.001)
        
        self.W_b = nn.Parameter(response)

    def forward(self, x):
        conv_flat = torch.reshape(x, [-1, px_conv, channels, 1])
        W_spatial_flat = torch.reshape(self.W_spatial, [px_conv, 1, 1, num_neurons])
        h_spatial = torch.nn.functional.conv2d(conv_flat, W_spatial_flat)
        self.h_out = torch.sum(h_spatial * self.W_features, dim = (1, 2))
        
        return self.h_out + self.W_b


lamd_s, lamd_d = 0.1, 0.1
def L_e(y,pred):
    return torch.mean(torch.sqrt(torch.sum((y-pred)**2,dim=1)))

def L_2(W_spatial,W_features,lamd_s=lamd_s,lamd_d=lamd_d):
    return lamd_s * torch.sum(W_spatial**2) + lamd_d * torch.sum(W_features**2)

K = torch.tensor([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]],dtype=torch.float).to(device)
def L_laplace(W_spatial,lamd_s=lamd_s):
    return lamd_s * torch.sum(F.conv2d(torch.unsqueeze(W_spatial,1),K.unsqueeze(0).unsqueeze(0))**2)

def train_model(encoder, optimizer):
    losses = []
    encoder.train()
    for i,(x,y) in enumerate(loader_train):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        out = encoder(x)
        l_e = L_e(y,out)
        l_2 = L_2(encoder.W_s,encoder.W_d)
        l_l = L_laplace(encoder.W_s)
#         print(f'L_e = {l_e} , L_2 = {l_2} , L_l = {l_l}')
        loss = L_e(y,out) + L_2(encoder.W_s,encoder.W_d) + L_laplace(encoder.W_s)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
#         print(f'iteration {i}, train loss: {losses[-1]}')
    
    return losses

def validate_model(encoder):
    encoder.eval()
    y_pred = []
    y_true = []
    losses = []
    for i,(x,y) in enumerate(loader_val):
        x = x.to(device)
        y = y.to(device)
        out = encoder(x)
        y_pred.append(out)
        y_true.append(y)
        l_e = L_e(y,out)
        l_2 = L_2(encoder.W_s,encoder.W_d)
        l_l = L_laplace(encoder.W_s)
        print(f'L_e = {l_e} , L_2 = {l_2} , L_l = {l_l}')
        loss = L_e(y,out) + L_2(encoder.W_s,encoder.W_d) + L_laplace(encoder.W_s)
        losses.append(loss.item())
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    explained_variance = metrics.explained_variance_score(y_true = y_true.detach().cpu().numpy(),y_pred = y_pred.detach().cpu().numpy())
    return explained_variance,sum(losses)/len(losses)

lr = 1e-3
encoder = conv_encoder(neurons, sizes, channels).to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

for epoch in tqdm(range(epoches)):
    losses_train += train_model(encoder,optimizer)
    ev,loss = validate_model(encoder)
    EVs.append(ev)
    losses_val.append(loss)
    print(f'epoch {epoch}, EV = {ev}, val  loss = {loss} , train loss {sum(losses_train[-10:])/10}')