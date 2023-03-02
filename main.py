import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math

def smoothness_regularizer_2d(W):
    with torch.autograd.profiler.record_function('smoothness'):
        lap = torch.tensor([[0., -1., 0.], [-1., 4.0, -1.], [0., -1., 0.]])
        lap = lap.unsqueeze(0).unsqueeze(0)
        out_channels = W.shape[2]
        W_lap = torch.nn.functional.conv2d(W.permute(3, 0, 1, 2),
                                           lap.repeat(1, out_channels, 1, 1),
                                           stride=1, padding=1, groups=out_channels)

        penalty = torch.mean(torch.sqrt(torch.sum(W_lap**2, dim=[1, 2])))
        return penalty

def size_regularizer_2d(W, circmask):
    with torch.autograd.profiler.record_function('size'):
        circmask = torch.tensor(circmask.astype('float32'))
        circmask = circmask.unsqueeze(0).unsqueeze(0)
        rf_mask = torch.mul(W, circmask)
        penalty = torch.mean(torch.sqrt(torch.sum(rf_mask**2, dim=[1, 2])))
        return penalty

def l2_norm_regularizer(W):
    with torch.autograd.profiler.record_function('l2_norm'):
        penalty = torch.mean(torch.sum(W**2, dim=0))
        return penalty

def l1_norm_regularizer(W):
    with torch.autograd.profiler.record_function('l1_norm'):
        penalty = torch.mean(torch.sum(torch.abs(W), dim=0))
        return penalty

def group_sparsity_regularizer_2d(W):
    with torch.autograd.profiler.record_function('group_sparsity'):
        penalty = torch.sum(torch.sqrt(torch.sum(W**2, dim=[0, 1])))
        return penalty

def elu(x):
    return torch.log(torch.exp(x) + 1)

def inv_elu(x):
    return torch.log(torch.exp(x) - 1)

def poisson(prediction, response):
    return torch.mean(torch.sum(prediction - response * torch.log(prediction + 1e-5), dim=1))

def mse_loss(prediction, response, weight=None):
    if weight is None:
        mse_loss = torch.mean(torch.mean((prediction - response)**2, dim=0))
    else:
        mse_loss = torch.sum(weight*torch.mean((prediction - response)**2, dim=0))
    return mse_loss

def create_cell_weight(pcc, threshold=0.29):
    one = torch.ones_like(pcc, dtype=torch.float32)
    zero = torch.zeros_like(pcc, dtype=torch.float32)
    middle = torch.where(pcc<threshold, pcc/threshold, one)
    weight = torch.where(pcc>=0, middle, zero)
    weight_normed = torch.div(weight, torch.sum(weight))
    # if pcc >= threshold 1
    # if pcc < 0          0
    # if 0 <= pcc < threshold, pcc/threshold
    return weight_normed

def explained_variance_score(prediction, response):
    num = torch.mean((prediction - response)**2, dim=0)
    den = torch.var(response, unbiased=False, dim=0)
    ve = 1 - num* (1/den)
    ve_avg = torch.mean(ve)
    return ve_avg


def torch_pearson(prediction, response):
    prediction_mean = torch.mean(prediction, dim=0)
    response_mean = torch.mean(response, dim=0)
    num = torch.sum((prediction - prediction_mean) * (response - response_mean), dim=0)
    den = torch.sqrt(torch.sum((prediction - prediction_mean) ** 2, dim=0) *
                     torch.sum((response - response_mean) ** 2, dim=0))
    return num * (1 / den)


def torch_pearson_mat(input):
    input_mean = torch.mean(input, dim=0)
    num = torch.matmul(torch.transpose(input - input_mean), (input - input_mean))
    den = torch.matmul(torch.transpose(torch.sqrt(torch.sum((input - input_mean) ** 2, dim=0, keepdim=True))),
                       torch.sqrt(torch.sum((input - input_mean) ** 2, dim=0, keepdim=True)))
    return torch.mul(num, (1 / den))


def rf_sim_regularizer(W, loc_weight):
    penalty = torch.sum(torch.mul(W, torch.tensor(loc_weight, dtype=torch.float32)))
    return penalty


def creat_circmask(H, W):
    mask = torch.zeros((H, W))
    x, y = torch.meshgrid(torch.arange(H), torch.arange(W))
    cx, cy = H / 2, W / 2
    radius = int(1.0 * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)

    circmask = r2 >= radius * radius
    mask[circmask] = ((torch.sqrt(r2) - radius + 1e-8) / (H / 2 - radius + 1e-8))[circmask]
    return mask

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
        mask = creat_circmask(sizes[0], sizes[1])
        out = torch.stack(
            [F.conv2d(out[:,n,:,:,:],torch.unsqueeze(self.W_d[n],0)) for n in range(neurons)],dim=1)
            #dimension:N,n,1,h,w
        out = torch.sum(out,dim=(2,3,4))
        out = out + self.W_b
        return out

def L_e(y,pred):
    return torch.mean(torch.sqrt(torch.sum((y-pred)**2,dim=1)))

def L_2(W_s,W_d,lamd_s=lamd_s,lamd_d=lamd_d):
    return lamd_s * torch.sum(W_s**2) + lamd_d * torch.sum(W_d**2)

K = torch.tensor([
    [0,-1,0],
    [-1,4,-1],
    [0,-1,0]],dtype=torch.float).to(device)
def L_laplace(W_s,lamd_s=lamd_s):
    return lamd_s * torch.sum(F.conv2d(torch.unsqueeze(W_s,1),K.unsqueeze(0).unsqueeze(0))**2)


#encoder = conv_encoder(neurons, sizes, channels).to(device)
encoder = conv_encoder(neurons, sizes, channels).to(device)