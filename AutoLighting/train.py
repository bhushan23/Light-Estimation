import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
import torchvision
import os
import matplotlib
matplotlib.use('agg')
import pickle
import copy
import h5py
import pandas as pd
import random
import os
from utils import *
import dataLoading
import utils
from lossfunctions import *
from shading import *
import models
from utils import PRINT

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!


# Feature Net Training
def feature_net_train(fNet, lNet, image, label, output_path = '', normal = None, fixed_input = None, fixed_mask = None, training_real = False, num_epochs = 3):
    fOpt = torch.optim.Adam(fNet.parameters(), lr = 0.0002)
    lOpt = torch.optim.Adam(lNet.parameters(), lr = 0.0002)
    
    for epoch in range(0, num_epochs):
        tLoss = 0
        for s1, l in zip(image, label):
            #print s1.type
            #print 's1', s1.shape
            batchSize = s1.shape[0]
            #print batchSize
            s1 = var(s1).type(dtype)
            l = var(l).type(dtype)
            #s1 = s1.transpose(1, 3)
            output = fNet(s1)
            output = lNet(output)
            Floss = regression_loss(output, l)
            fNet.zero_grad()
            lNet.zero_grad()
            Floss.backward()
            fOpt.step()
            lOpt.step()
            tLoss += Floss
        print('Epoch:', epoch, 'Loss:', tLoss.data[0])
        if training_real == True:
            fixedSH = lNet(fNet(fixed_input))
            outShadingB = ShadingFromDataLoading(fixed_input, fixedSH, shadingFromNet = True)
            outShadingB = denorm(outShadingB)
            outShadingB = applyMask(outShadingB, fixed_mask)
            outShadingB = outShadingB.data
            save_image(outShadingB, output_path+'images/image_{}.png'.format(epoch))
    # No need to return
    # return fNet, lNet

def predict(fNet, lNet, input):
    val = next(iter(input)).type(torch.FloatTensor)
    val = var(val)
    out = lNet(fNet(val))
    return out

def predict_lighting_features(fNet, data):
    First = True

    for s1 in data:
        s1 = var(s1)
        out = fNet(s1)
        if First == True:
            fsFeatures = out.data
            First = False
        else:
            # print(fsFeatures.data.shape, out.data.shape)
            fsFeatures = torch.cat((fsFeatures, out.data), dim = 0)
    return fsFeatures

# Use this to denoise SH
def denoised_SH(vae, fNet, images, noisy_sh, batch_size):
    lighting_features = predict_lighting_features(fNet, images)
    # lighting_features = lighting_features
    lighting_features = torch.utils.data.DataLoader(lighting_features, batch_size= batch_size, shuffle = False)

    First = True
    for l_feature, n_sh in zip(lighting_features, noisy_sh):
        n_sh = n_sh.type(dtype)
        input = torch.cat((l_feature, n_sh), dim = 1)
        input = var(input)
        output, mu, log_var = vae(input)
        _, d_sh = torch.split(output, 128, dim = 1)
        if First:
            denoised_sh = d_sh
            First = False
        else:
            denoised_sh = torch.cat((denoised_sh, d_sh), dim = 0)
    return denoised_sh

# Training Variational AE
def trainVAE(vNet, fNet, images, noisy_sh, true_sh, batch_size = 64, num_epochs = 10):
    vNet_opt = torch.optim.Adadelta(vNet.parameters(), lr = 0.0002)

    lighting_features = predict_lighting_features(fNet, images)
    lighting_features = torch.utils.data.DataLoader(lighting_features, batch_size= batch_size, shuffle = False)
    
    for epoch in range(0, num_epochs):
        for l_feature, n_sh, t_sh in zip(lighting_features, noisy_sh, true_sh):
            n_sh = n_sh.type(dtype)
            t_sh = t_sh.type(dtype)
            input = torch.cat((l_feature, n_sh), dim = 1)
            expected_output = var(torch.cat((l_feature, t_sh), dim = 1).type(dtype))
             
            input = var(input)
            output, mu, log_var = vNet(input)

            reconst_loss = F.binary_cross_entropy(output, expected_output, size_average=False)
            kl_divergence = torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var -1))

            # Backprop + Optimize
            total_loss = reconst_loss + kl_divergence
            vNet_opt.zero_grad()
            total_loss.backward()
            vNet_opt.step()

        print 'Epoch [{}/{}], VAE Loss: {}'.format(epoch+1, num_epochs, total_loss.data[0])

        if epoch+1 % 100 == 0:
            torch.save(vNet.state_dict(), output_path+'savedModels/vNet_'+str(epoch/100)+'.pkl')
