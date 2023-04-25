# 模型
import os
import pickle
import sys
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import numpy as np
from tqdm import tnrange, tqdm_notebook
import models
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import h5py
from PIL import Image

device = 'cuda:0' # device where you put your data and models
data_path = './' # the path of the 'npc_v4_data.h5' file
batch_size = 16 # the batch size of the data loader
insp_layer = 'conv3' # the middle layer extracted from alexnet, available in {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}

mse_weight = 1.0
l1_weight = 0
spa_weight = 1e-1
ch_weight = 1e-1
lap_weight = 1e-1


#定义损失函数
K = torch.tensor([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]],dtype=torch.float)
def mse_loss(prediction, response, weight=None):
    if weight is None:
        mse_loss = torch.mean(torch.mean((prediction - response)**2, dim=1))
    else:
        mse_loss = torch.sum(weight*torch.mean((prediction - response)**2, dim=1))
    return mse_loss

def l2_norm_regularizer(W):
    return torch.mean(torch.sum(W**2))

def l1_norm_regularizer(W):
    return torch.mean(torch.sum(torch.abs(W)))

def smoothness_regularizer_2d(W_s):
    global K
    return torch.sum(F.conv2d(torch.unsqueeze(W_s,1),K.unsqueeze(0).unsqueeze(0))**2)
def torch_pearson(prediction, response):
    prediction_mean = torch.mean(prediction, dim=0)
    response_mean = torch.mean(response, dim=0)
    num = torch.sum((prediction - prediction_mean)*(response - response_mean), dim=0)
    den = torch.sqrt(torch.sum((prediction - prediction_mean)**2, dim=0) *
                     torch.sum((response - response_mean)**2, dim=0))
    return num * (1/den)
def explained_variance_score(prediction, response):
    num = torch.mean((prediction - response)**2, dim=0)
    den = torch.var(response, dim=0)
    ve = 1 - num* (1/den)
    ve_avg = torch.mean(ve)
    return ve_avg

class conv_encoder(nn.Module):

    def __init__(self, neurons, sizes, channels, reg_model_weight = None):
        super(conv_encoder, self).__init__()
        # PUT YOUR CODES HERE
        self.neurons = neurons
        self.channels = channels
        sz = (self.sizes[0], self.sizes[1], self.channels)
        self.px_x_conv = int(sz[2])
        self.px_y_conv = int(sz[1])
        self.px_conv = self.px_x_conv * self.px_y_conv

        if reg_model_weight is not None:
            ws_initial_value = torch.from_numpy(reg_model_weight['W_s'][:].reshape(self.neurons, self.px_conv)).transpose(0, 1).float()
            self.W_spatial = torch.nn.Parameter(ws_initial_value)
        else:
            self.W_spatial = torch.nn.Parameter(torch.randn(self.px_conv, neurons) * 0.001)

        if reg_model_weight is not None:
            wf_initial_value = torch.from_numpy(reg_model_weight['W_d'][:]).transpose(0, 1).float()
            self.W_features = torch.nn.Parameter(wf_initial_value)
        else:
            self.W_features = torch.nn.Parameter(torch.randn(int(sz[3]), neurons) * 0.001)
        if reg_model_weight is not None:
            b_initial_value = torch.from_numpy(reg_model_weight['W_b'][:]).float()
            self.W_b = torch.nn.Parameter(b_initial_value)
        else:
            self.W_b = torch.nn.Parameter(torch.zeros(neurons))

    def forward(self, x):
        # PUT YOUR CODES HERE
        self.fts = x
        conv_flat = torch.reshape(self.fts, (-1, self.px_conv, int(self.channels), 1)) # [batch, 17 * 17, 384]
        W_spatial_flat = torch.reshape(self.W_spatial, [self.neurons, self.px_conv, 1, 1]) # [43, 17 * 17, 1, 1]
        conv_flat = conv_flat.to(device)
        W_spatial_flat = W_spatial_flat.to(device)
        h_spatial = F.conv2d(conv_flat, W_spatial_flat, stride=1, padding=0)
        h_out = torch.sum(torch.mul(h_spatial, self.W_features), dim=[1, 2])
        return h_out + self.W_b

def Loss(y, pred, W_s, W_d):
    return mse_loss(y, pred) * mse_weight + \
          l2_norm_regularizer(W_s) * spa_weight + \
          smoothness_regularizer_2d(W_s) * lap_weight + \
          l2_norm_regularizer(W_d) * ch_weight


def Loss(y, pred, W_s, W_d):
    return mse_loss(y, pred) * mse_weight + \
          l2_norm_regularizer(W_s) * spa_weight + \
          smoothness_regularizer_2d(W_s) * lap_weight + \
          l2_norm_regularizer(W_d) * ch_weight


device = 'cuda:0' # device where you put your data and models
data_path = './' # the path of the 'npc_v4_data.h5' file
batch_size = 16 # the batch size of the data loader
insp_layer = 'conv3' # the middle layer extracted from alexnet, available in {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}


# image_data
root_dir = '../data/0_presented_images_800/'
resolution = 300
image_path = os.listdir(root_dir)
path_dict = {}
for j in image_path:
    key = int(j.split('_')[0])  # 刺激呈现的顺序是图像名称下划线前面的数字顺序。
    path_dict[key] = j

stim_arr = np.zeros((len(image_path), resolution, resolution, 3))
for i in range(len(image_path)):
    img_bgr = cv2.imread(os.path.join(root_dir, path_dict[i+1]))
    stim_arr[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
stim_arr = stim_arr.astype('float32')

# random repeat
id = h5py.File('../data/1_L76LM_V1_S18_D155_objects/stimuli/Random_id_80_2021_10_21.mat', 'r')

images_n  = np.zeros(shape=(stim_arr.shape[0], 299, 299, 3))
for i in range(stim_arr.shape[0]):
    images_n[i] = cv2.resize(stim_arr[i], (299, 299))

idx = np.array(id['sampleidlist21']).squeeze().astype('int') - 1
print(idx)
idx, unique_idx = np.unique(idx, return_index=True)
print(idx, unique_idx, images_n.shape)

# neurons_data
mat_file = h5py.File('../data/1_L76LM_V1_S18_D155_objects/celldataS_43_Objects_11_800_80_30_40_trial_mean_normal.mat', 'r')
#[num_repetitions, num_images, num_neurons]
#print(np.array(mat_file['celldataS']).shape)
neural_n = np.transpose(np.array(mat_file['celldataS']), (1, 2, 0)).astype('float16')
neural_n = neural_n[:,:880, :]
print(neural_n.shape)
#12个trials 880张图片（其中80张是重复），114个细胞

n_images = 800
n_neurons = neural_n.shape[2]
size_imags = images_n.shape[0]
print(n_images, n_neurons, images_n.shape)
#encoder = conv_encoder(neurons, sizes, channels).to(device)
