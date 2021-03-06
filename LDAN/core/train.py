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


# Synthetic Net Training
def syn_net_train(fNet, lNet, syn_image1, syn_image2, syn_label, num_epochs = 3):
    fOpt = torch.optim.Adam(fNet.parameters(), lr = 0.001)
    lOpt = torch.optim.Adam(lNet.parameters(), lr = 0.001)

    for epoch in range(0, num_epochs):
        tLoss = 0
        for s1, s2, l in zip(syn_image1, syn_image2, syn_label):
            #print s1.type
            #print 's1', s1.shape
            batchSize = s1.shape[0]
            #print batchSize
            s1 = var(s1).type(dtype)
            s2 = var(s2).type(dtype)
            l = var(l).type(dtype)
            #s1 = s1.transpose(1, 3)
            output = fNet(s1)
            output = lNet(output)
            output2 = fNet(s2)
            output2 = lNet(output2)
            Floss = regression_loss(output, output2, l)
            fNet.zero_grad()
            lNet.zero_grad()
            Floss.backward()
            fOpt.step()
            lOpt.step()
            tLoss += Floss
        print('Epoch:', epoch, 'Total Loss:', tLoss.data[0], 'Loss:', Floss.data[0])
        if epoch+1 % 50 == 0:
            torch.save(fNet.state_dict(), output_path+'savedModels/fNet_'+str(epoch/100)+'.pkl')
            torch.save(lNet.state_dict(), output_path+'savedModels/lNet_'+str(epoch/100)+'.pkl')


    # No need to return
    # return fNet, lNet

def predict(fNet, lNet, input):
    val = next(iter(input)).type(torch.FloatTensor)
    val = var(val)
    out = lNet(fNet(val))
    return out

def predictAllSynthetic(fNet, data):
    fsFeatures = []
    i = 0
    for s1 in data:
        s1 = var(s1)
        fsFeatures.append(fNet(s1))
        i += 1
        if i == 10:
            break
    return fsFeatures

# Training GAN
def trainGAN(lNet, rNet, D, featureNet, syn_image1, rData, rLabel, fixed_input, sirfs_fixed_normal, real_image_mask, fixed_label, fixed_sirfs_label = None, output_path = './', numDTrainer= 1, numGTrainer = 1, num_epoch = 5):
    rNet_opt = torch.optim.Adadelta(rNet.parameters(), lr = 0.001)
    #lNet_opt = torch.optim.Adadelta(lNet.parameters(), lr = 0.0002)
    D_opt    = torch.optim.RMSprop(D.parameters(), lr = 0.001)

    syn_image_iter = iter(syn_image1)
    syn_image_len  = len(syn_image_iter)
    syn_image_cnt  = 0
    firstCallD = False
    firstCallG = False
    mse_sirfs_error = []
    mse_error = []
    fixed_sirfs_label = fixed_sirfs_label.type(dtype)
    fixed_label = fixed_label.type(dtype)
    for epoch in range(0, num_epoch):
        GLoss_D = 0.0
        DLoss_D = 0.0
        for rD, rL in zip(rData, rLabel):
            image = var(rD).type(dtype)
            #print type(image)
            rL = var(rL).type(dtype)
            batch_size = image.size(0)
            #print batch_size
            # Train the Discriminator
            for k in range(0, numDTrainer):
                # Randomly pick fs to train Discriminator
                #print fs[rFS].shape
                # Get Lighting for Synthetic image
                if syn_image_cnt == syn_image_len:
                    syn_image_iter = iter(syn_image1)
                    syn_image_cnt = 0
                syn_image_cnt += 1
                s1 = var(next(syn_image_iter))
                fs = featureNet(s1)

                D_real = D(fs)
                #print(D_real.size())
                #print(image.size())
                # Pass real data through generator
                G_fake = rNet(image)
                D_fake = D(G_fake)
                # Loss for Discriminator
                #D_real_loss = lossGAN(D_real, var(torch.ones(batch_size, 1)))
                #D_fake_loss = lossCriterion(D_fake, var(-1 * torch.ones(batch_size, 1)))
                D_real_loss = GAN_loss(D_real)
                D_fake_loss = GAN_loss(D_fake)
                #print 'DLOSS:', D_real_loss.data[0], ' ', D_fake_loss.data[0]
                D_loss = -D_real_loss + D_fake_loss # -ve as we need to maximize

                # Backprop Discriminator
                D.zero_grad()
                if firstCallD == True:
                    D_loss.backward(retain_graph=True)
                    firstCallD = False
                else:
                    D_loss.backward(retain_graph = True)
                D_opt.step()

            # Train the Generator
            for k in range(0, numGTrainer):
                G_fake = rNet(image)
                D_fake = D(G_fake)
                # Generator Loss
                G_predict = lNet(G_fake)
                #print type(G_predict), type(rL)
                #G_loss = lossCriterion(D_fake, var(torch.ones(batch_size, 1))) + MU * regressionLossSynthetic(G_predict, rLabel)
                G_loss = -GAN_loss(D_fake) + MU * regression_loss_synthetic(G_predict, rL).sum()
                #G_loss = MU * regressionLossSynthetic(G_predict, rL).sum()

                #lNet.zero_grad()
                rNet.zero_grad()
                if firstCallG == True:
                    G_loss.backward(retain_graph=True)
                    firstCallG = False
                else:
                    G_loss.backward(retain_graph = True)
                rNet_opt.step()
                #lNet_opt.step()
            GLoss_D += G_loss.data[0]
            DLoss_D += D_loss.data[0]

        print 'Epoch [{}/{}], Discriminator {}|{}, Generator {}|{}'.format(epoch+1, num_epoch, D_loss.data[0], DLoss_D, G_loss.data[0], GLoss_D)
        # print('I Size:', fixed_input.data.size())
        fixedSH = lNet(rNet(fixed_input))
        out_sh = fixedSH.type(dtype)
        mse = feature_loss(out_sh.data, fixed_label.data)
        mse = mse.mean(dim = 0)
        mse = mse.mean()
        mse_error.append(mse)
        mse = feature_loss(out_sh.data, fixed_sirfs_label.data)
        mse = mse.mean(dim = 0)
        mse = mse.mean()
        mse_sirfs_error.append(mse)

        # print('OUTPUT OF fixedSH:', fixedSH.data.size(), sirfs_fixed_normal.size())
        outShadingB = ShadingFromDataLoading(sirfs_fixed_normal, fixedSH, shadingFromNet = True)
        outShadingB = denorm(outShadingB)
        # if real_image_mask != None:
        outShadingB = applyMask(outShadingB, real_image_mask)
        outShadingB = outShadingB.data
        #pic = torchvision.utils.make_grid(outShadingB, padding=1)
        save_image(outShadingB, output_path+'images/image_{}.png'.format(epoch))

        if epoch+1 % 100 == 0:
            torch.save(lNet.state_dict(), output_path+'savedModels/GAN_LNet_'+str(epoch/100)+'.pkl')
            torch.save(D.state_dict(), output_path+'savedModels/Discriminator_'+str(epoch/100)+'.pkl')
            torch.save(rNet.state_dict(), output_path+'savedModels/Generator_'+str(epoch/100)+'.pkl')
    plt.title('MSE Loss on Validation')
    plt.plot(mse_error, label = 'Ground Truth SH')
    plt.plot(mse_sirfs_error, label = 'SIRFS SH')
    plt.legend()
    plt.savefig(output_path+'Validation_MSE')

