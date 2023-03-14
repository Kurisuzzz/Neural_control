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

root_dir = 'C:/Users/admin/Desktop/pytorch_ovo/data/0_presented_images_800/'
resolution = 300
image_path = os.listdir(root_dir)
path_dict = {}
for j in image_path:
    key = int(j.split('_')[0])  # 刺激呈现的顺序是图像名称下划线前面的数字顺序。
    path_dict[key] = j

stim_arr = np.zeros((len(image_path), resolution, resolution, 3))
# stim_arr_gray3 = np.zeros((len(image_path), resolution, resolution, 3))
for i in range(len(image_path)):
    img_bgr = cv2.imread(os.path.join(root_dir, path_dict[i+1]))
    stim_arr[i] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
stim_arr = stim_arr.astype('float32')


id = h5py.File('C:/Users/admin/Desktop/pytorch_ovo/data/4_L82LL_V4_S03_D250_objects/stimuli/Random_id_80_2021_10_21.mat', 'r')

images_n  = np.zeros(shape=(stim_arr.shape[0], 299, 299, 3))
for i in range(stim_arr.shape[0]):
    images_n[i] = cv2.resize(stim_arr[i], (299, 299))

idx = np.array(id['sampleidlist21']).squeeze().astype('int') - 1
print(idx)
idx, unique_idx = np.unique(idx, return_index=True)
print(idx, unique_idx, images_n.shape)
mat_file = h5py.File('C:/Users/admin/Desktop/pytorch_ovo/data/4_L82LL_V4_S03_D250_objects/stimuli/celldataS_80_CalmAn_75_Objects_16_800_80_50_88_62_trial_mean_normal.mat', 'r')
#[num_repetitions, num_images, num_neurons]
#print(np.array(mat_file['celldataS']).shape)
neural_n = np.transpose(np.array(mat_file['celldataS']), (1, 2, 0)).astype('float16')
neural_n = neural_n[:,:880, :]
print(neural_n.shape)
n_images = 800
n_neurons = neural_n.shape[2]
size_imags = images_n.shape[0]
print(n_images, n_neurons, images_n.shape)
reps = neural_n.shape[0] # trials
rand_ind = np.arange(reps)
np.random.shuffle(rand_ind)

data_y_train = np.concatenate((np.delete(neural_n[:, :800, :], idx, 1), neural_n[:, 880:, :]), 1).mean(0)
temp = np.transpose(neural_n, (1, 0, 2))
print(temp.shape, idx.shape, temp[idx].shape)
data_y_val = np.concatenate((temp[idx], temp[800:880][unique_idx]), 1)
data_y_val = np.transpose(data_y_val, (1, 0, 2))
data_y_val = np.mean(data_y_val, 0)
print(data_y_train.shape)
print(data_y_val.shape)

#
# data_x = images_n[:, np.newaxis].astype(np.float16)
# print('images_n', images_n.shape)
# data_x = data_x / 255 # (640, 1, 299, 299)
# data_x = np.tile(data_x, [1, 3, 1, 1])
# print('data_x', data_x.shape)
# data_x_train = data_x[:576]
# data_x_val = data_x[576:]as indices must be
print(images_n.shape)
#data_x = images_n[:, np.newaxis].astype(np.float16)
data_x = images_n.astype(np.float16)
print(data_x.shape)
data_x = data_x / 255 # (800, 1, 299, 299)

#data_x = np.tile(data_x, [1, 3, 1, 1])
data_x_train = np.delete(images_n, idx, 0)
data_x_val = images_n[idx]

data_x = np.transpose(data_x, (0, 3, 1, 2))
data_x_train = np.transpose(data_x_train, (0, 3, 1, 2))
data_x_val = np.transpose(data_x_val, (0, 3, 1, 2))
print(data_x.shape, data_x_train.shape, data_x_val.shape)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
    def __getitem__(self, index):
        return index, self.data_x[index], self.data_y[index]
    def __len__(self):
        return self.data_x.shape[0]



dataset_train = Dataset(data_x_train, data_y_train)
dataset_val = Dataset(data_x_val, data_y_val)

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle = True)

print(f'val: {data_x_val.shape}, {data_y_val.shape}')
# for i,(x,y) in enumerate(loader_val):
#     print(i, x.shape, y.shape)
# CHOOSE THE AUGMENTS IF NECESSARY


alexnet = models.alexnet(pretrained=True)

#
alexnet.to(device)
alexnet.eval()
for param in alexnet.parameters():
    param.requires_grad_(False)

x = torch.from_numpy(data_x[0:1]).to(device)
print("x:", x.shape)
x = x.float()
fmap = alexnet(x, layer=insp_layer)

neurons = data_y_train.shape[1]
sizes = fmap.shape[2:]
print("fmap: ", fmap.shape)
print("size: ", sizes)
channels = fmap.shape[1]
print(neurons, sizes)

x = torch.from_numpy(data_x[0:1]).float().to(device)
fmap = alexnet(x, layer=insp_layer)
print(fmap.shape)
sizes = fmap.shape[2:]

imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).to(device)
transform = lambda x : (x - imagenet_mean) / imagenet_std
#data_x = transform(data_x)
feature_map = torch.Tensor(n_images, fmap.shape[1], fmap.shape[2], fmap.shape[3])
feature_map.to(device)
print(feature_map.shape)
for i in range(n_images):
    x = torch.from_numpy(data_x[i:i + 1])
    x = transform(x).to(device)
    fmap = alexnet(x, layer = insp_layer)
    feature_map[i] = fmap
mse_weight = 1.0
l1_weight = 0
spa_weight = 1e-1
ch_weight = 1e-1
lap_weight = 1e-1
def mse_loss(prediction, response, weight=None):
    if weight is None:
        mse_loss = torch.mean(torch.mean((prediction - response)**2, dim=1))
    else:
        mse_loss = torch.sum(weight*torch.mean((prediction - response)**2, dim=1))
    return mse_loss

def l2_norm_regularizer(W):
    with torch.autograd.profiler.record_function('l2_norm'):
        penalty = torch.mean(torch.sum(W**2))
        return penalty

def l1_norm_regularizer(W):
    with torch.autograd.profiler.record_function('l1_norm'):
        penalty = torch.mean(torch.sum(torch.abs(W)))
        return penalty

def smoothness_regularizer_2d(W_s):
    K = torch.tensor([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]],dtype=torch.float).to(device)
    return torch.sum(F.conv2d(torch.unsqueeze(W_s,1),K.unsqueeze(0).unsqueeze(0))**2)

def pearson_corr(prediction, response):
    prediction = torch.transpose(prediction, 1, 0)
    response = torch.transpose(response, 1, 0)

    prediction_mean = torch.mean(prediction, dim=0)
    response_mean = torch.mean(response, dim=0)

    num = torch.sum((prediction - prediction_mean)*(response - response_mean), dim=0)
    den = torch.sqrt(torch.sum((prediction - prediction_mean)**2, dim=0) *
                     torch.sum((response - response_mean)**2, dim=0))
    pcc = torch.mean(num * (1 / den))
    return pcc

def explained_variance(y_true, y_pred):
    total_var = torch.var(y_true)
    residual_var = torch.var(y_true - y_pred)
    explained_var = total_var - residual_var
    return explained_var.item()

class conv_encoder(nn.Module):

    def __init__(self, neurons, sizes, channels):
        super(conv_encoder, self).__init__()
        # PUT YOUR CODES HERE
        self.W_s = nn.Parameter(torch.randn(size=(neurons,) + sizes))
        self.W_d = nn.Parameter(torch.randn(size = (neurons,channels,1,1)))
        self.W_b = nn.Parameter(torch.randn(size = (1,neurons)))


    def forward(self, x):
        # PUT YOUR CODES HERE
        out = torch.einsum('bchw , nhw -> bnchw',x,self.W_s) # dimension : N,n,C,h,w
        out = torch.stack(
            [F.conv2d(out[:,n,:,:,:],torch.unsqueeze(self.W_d[n],0)) for n in range(neurons)],dim=1)
            #dimension:N,n,1,h,w
        out = torch.sum(out,dim=(2,3,4))
        out = out + self.W_b
        return out

def Loss(y, pred, W_s, W_d):
    return mse_loss(y, pred) * mse_weight + \
          l2_norm_regularizer(W_s) * spa_weight + \
          smoothness_regularizer_2d(W_s) * lap_weight + \
          l2_norm_regularizer(W_d) * ch_weight




#encoder = conv_encoder(neurons, sizes, channels).to(device)
encoder = conv_encoder(neurons, sizes, channels).to(device)

def train_model(encoder, optimizer):
    losses = []
    encoder.train()
    for i,(z, x,y) in enumerate(loader_train):
        optimizer.zero_grad()
        x = x.float().to(device)
        y = y.float().to(device)
        z = z.to(device)
        x = transform(x)
        fmap = feature_map[z - 1].to(device)
        out = encoder(fmap)
#         print(f'L_e = {l_e} , L_2 = {l_2} , L_l = {l_l}')
        loss = Loss(y, out, encoder.W_s, encoder.W_d)
        loss.backward()
        del loss
        optimizer.step()
        losses.append(loss.item())
#         print(f'iteration {i}, train loss: {losses[-1]}')

    return losses

def validate_model(encoder):
    encoder.eval()
    y_pred = []
    y_true = []
    losses = []
    for i,(z, x,y) in enumerate(loader_val):
        x = x.float().to(device)
        y = y.float().to(device)
        z = z.to(device)
        x = transform(x).to(device)
        fmap = alexnet(x, layer=insp_layer)
        #x = transform(x)
        #fmap = feature_map[z - 1].to(device)
        out = encoder(fmap)
        y_pred.append(out)
        y_true.append(y)
        loss = Loss(y, out, encoder.W_s, encoder.W_d)
        losses.append(loss.item())
        del loss
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    ev = explained_variance(y_true, y_pred)
    pcc = pearson_corr(y_pred, y_true)
    return pcc, ev,sum(losses)/len(losses)
    #return explained_variance,sum(losses)/len(losses)

"""
    You need to define the conv_encoder() class and train the encoder.
    The code of alexnet has been slightly modified from the torchvision, for convenience
    of extracting the middle layers.

    Example:
        >>> x = x.to(device) # x is a batch of images
        >>> x = transform(x)
        >>> fmap = alexnet(x, layer=insp_layer)
        >>> out= encoder(fmap)
        >>> ...
"""
# losses_train = []
# losses_val = []
# EVs = []

losses_train = []
losses_val = []
EVs = []
pccs = []
lr = 1e-2
optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
#optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
epoches = 2000
best_loss = 1e100
not_improve = 0
endure = 10
for epoch in tqdm_notebook(range(epoches)):
    losses_train += train_model(encoder,optimizer)
    pcc, ev,loss = validate_model(encoder)
    #ev,loss = validate_model(encoder)
    EVs.append(ev)
    pccs.append(pcc)
    losses_val.append(loss)
    train_loss = sum(losses_train[-10:])/10
    if train_loss < best_loss - 1e-5:
        not_improve = 0
    else:
        not_improve += 1
    if epoch % 1 == 0:
        print(f'epoch {epoch}, EV = {ev}, val  loss = {loss} , train loss {sum(losses_train[-10:])/10}, pcc = {pcc}')
        #print(f'epoch {epoch}, EV = {ev}, val  loss = {loss} , train loss {sum(losses_train[-10:])/10}')
    if not_improve == endure:
        print("Early stopping!")

    scheduler.step()

