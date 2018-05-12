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

# Custom import
import dataLoading
import utils
from lossfunctions import *
from shading import *
import models
from utils import PRINT
from train import *

SHOW_IMAGES = False
load_syn    = False #True
load_real   = False #True
load_AE     = False
train_syn   = True #False
train_real  = True #False
train_AE    = True #False
FirstRun    = False #True
real_epochs = 1
syn_epochs  = 1
vae_epochs  = 1

LOCAL_MACHINE = False
exp_name      = 'basicNet_AE'
output_path   = './basicNet_AE/'

synthetic_data_path = '/home/bsonawane/Thesis/LightEstimation/Light-Estimation/data/AE/syn/'  #'/home/bhushan/college/CV/Thesis/Projects/lightestimation/Light-Estimation/data/AE/syn/'
real_data_path      = '/home/bsonawane/Thesis/LightEstimation/Light-Estimation/data/AE/real/'   #'/home/bhushan/college/CV/Thesis/Projects/lightestimation/Light-Estimation/data/AE/real/'

if FirstRun == True:
    os.mkdir(output_path)
    os.mkdir(output_path+'images/')
    os.mkdir(output_path+'val/')
    os.mkdir(output_path+'models/')
    os.mkdir(output_path+'savedModels/')

batch_size = 64

# Helper routines
IS_CUDA = False
if torch.cuda.is_available():
    IS_CUDA = True

def save_shading(normal, sh, real_image_mask, path, name, shadingFromNet = False, Predicted = False):
    if Predicted == False:
        normal = denorm(normal)
    outShadingB = ShadingFromDataLoading(normal, sh, shadingFromNet = True)
    #if real_image_mask != None:
    if Predicted == True:
        outShadingB = denorm(outShadingB)
    outShadingB = applyMask(outShadingB, real_image_mask)
    outShadingB = outShadingB.data
    #pic = torchvision.utils.make_grid(outShadingB, padding=1)
    save_image(outShadingB, path + name+'.png')
    save_image(outShadingB[0], path + name+'_0.png')


def var(x):
    if IS_CUDA:
        x = x.cuda()
    return Variable(x)
# End of Helper routines

# Load synthetic dataset
#syn_image1, syn_image2, syn_label = dataLoading.load_synthetic_ldan_data(synthetic_image_dataset_path)
#real_image, sirfs_normal, sirfs_SH, sirfs_shading, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val = dataLoading.load_real_images_celebA(real_image_dataset_path, validation = True)
#real_image_mask = dataLoading.getMask(real_image_mask, global_batch_size)
#real_image, sirfs_normal, sirfs_SH, sirfs_shading, tNormal, real_image_mask, tSH, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val, true_normal_val, mask_val, true_lighting_val = dataLoading.load_SfSNet_data(sfs_net_path, validation = True, twoLevel = True)

syn_image, syn_normal, syn_sh, syn_shading, syn_mask, syn_sirfs_shading, syn_sirfs_normal, syn_sirfs_sh, syn_image_val, syn_normal_val, syn_lighting_val, syn_shading_val, syn_mask_val, syn_sirfs_shading_val, syn_sirfs_normal_val, syn_sirfs_sh_val  = dataLoading.load_SfSNet_data(synthetic_data_path)

real_image, real_normal, real_sh, real_shading, real_mask, real_sirfs_shading, real_sirfs_normal, real_sirfs_sh, real_image_val, real_normal_val, real_lighting_val, real_shading_val, real_mask_val, real_sirfs_shading_val, real_sirfs_normal_val, real_sirfs_sh_val  = dataLoading.load_SfSNet_data(real_data_path, validation = True)

# Transforms being used
# if SHOW_IMAGES:
'''
syn_image_mask_test = next(iter(mask_val))
utils.save_image(torchvision.utils.make_grid(syn_image_mask_test*255, padding=1), output_path+'images/MASK_TEST.png')

tmp = next(iter(syn_image1))
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/test_synthetic_img.png')

tmp = var(next(iter(syn_image_val)))
tmp = denorm(tmp)
print(tmp.data.shape)
tmp = applyMask(tmp, syn_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_syn_image.png')

tmp = var(next(iter(syn_normal_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, syn_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_syn_normal.png')


tmp = var(next(iter(sirfs_normal_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, syn_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_sirf_normal.png')

tmp = var(next(iter(sirfs_shading_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, syn_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_sirf_shading.png')


tmp = var(next(iter(syn_shading_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, syn_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data*255, padding=1), output_path+'images/test_syn_shading.png')
'''

## TRUE SHADING

'''
tmp = next(iter(sirfs_shading_val))
tmp = utils.denorm(tmp)
tmp = applyMask(var(tmp).type(torch.DoubleTensor), syn_image_mask_test)
tmp = tmp.data
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/Validation_SIRFS_SHADING.png')
'''

# featureNet = ResNet(BasicBlock, [2, 2, 2, 2], 27)
featureNet = models.BaseSimpleFeatureNet()
lightingNet = models.LightingNet()
featureNet_real = models.BaseSimpleFeatureNet()
lightingNet_real = models.LightingNet()
vae = models.VAutoEncoder()

print(featureNet)
print(lightingNet)
featureNet = featureNet.cuda()
lightingNet = lightingNet.cuda()
featureNet_real = featureNet_real.cuda()
lightingNet_real = lightingNet_real.cuda()
vae = vae.cuda()

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!

# Training with Synthetic Images
if load_syn:
    featureNet.load_state_dict(torch.load(output_path+'models/featureNet.pkl'))
    lightingNet.load_state_dict(torch.load(output_path+ 'models/lightingNet.pkl'))

if train_syn:
    feature_net_train(featureNet, lightingNet, syn_image, syn_sh, num_epochs = syn_epochs)
    # save_image(predict(featureNet, lightingNet, synVal1), outPath+'_Synthetic_Image.png')
    torch.save(featureNet.state_dict(), output_path+'models/featureNet.pkl')
    torch.save(lightingNet.state_dict(),output_path+ 'models/lightingNet.pkl')


fixed_input = var(next(iter(real_image_val))).type(dtype)
sirfs_fixed_normal = var(next(iter(real_sirfs_normal_val)))
true_fixed_normal = var(next(iter(real_normal_val)))
true_fixed_lighting = var(next(iter(real_lighting_val)))

#syn_image_mask = next(iter(syn_image_mask))
#utils.show(torchvision.utils.make_grid(utils.denorm(fixed_input), padding=1))

if load_AE:
    vae.load_state_dict(torch.load(output_path+'models/vae.pkl'))

if train_AE:
    trainVAE(vae, featureNet, syn_image, syn_sirfs_sh, syn_sh, num_epochs = vae_epochs)
    torch.save(vae.state_dict(), output_path+'models/vae.pkl')


denoised_sh = denoised_SH(vae, featureNet, real_image, real_sirfs_sh, batch_size)

denoised_sh = torch.utils.data.DataLoader(denoised_sh, batch_size= batch_size, shuffle = False)
# NOW, We have denoised SH for real images
# Training Synthetic Net with Real images and Real SH

# Training with Real Images
if load_real:
    featureNet.load_state_dict(torch.load(output_path+'models/featureNet_real.pkl'))
    lightingNet.load_state_dict(torch.load(output_path+ 'models/lightingNet_real.pkl'))

if train_real:
    feature_net_train(featureNet_real, lightingNet_real, real_image, denoised_sh, output_path, sirfs_fixed_normal, fixed_input, training_real = True, num_epochs = real_epochs)
    # save_image(predict(featureNet, lightingNet, synVal1), outPath+'_Synthetic_Image.png')
    torch.save(featureNet.state_dict(), output_path+'models/featureNet_real.pkl')
    torch.save(lightingNet.state_dict(),output_path+ 'models/lightingNet_real.pkl')


## TESTING
#fixedSH = lightingNet(R(fixed_input))
#fixedSH = fixedSH.type(torch.DoubleTensor)

## With SIRFS_NORMAL
save_shading(sirfs_fixed_normal, fixedSH, syn_image_mask_test, path = output_path+'val/', name = 'PREDICTED_SIRFS_NORMAL', shadingFromNet = True, Predicted = True)

## With True Normal
save_shading(true_fixed_normal, fixedSH, syn_image_mask_test, path = output_path+'val/', name = 'PREDICTED_TRUE_NORMAL', shadingFromNet = True, Predicted = True)

## EXPECTED with SIRFS NORMAL
save_shading(sirfs_fixed_normal, true_fixed_lighting, syn_image_mask_test, path = output_path+'val/', name = 'EXPECTED_SIRFS_NORMAL', shadingFromNet = True)

## EXPECTED with true normal
save_shading(true_fixed_normal, true_fixed_lighting, syn_image_mask_test, path = output_path+'val/', name = 'EXPECTED_TRUE_NORMAL', shadingFromNet = True)


## Save single image
#tmp = next(iter(sirfs_shading))
#utils.save_image(tmp[0], output_path+'val/shading.png')

'''
fSH = fixedSH.cpu().data.numpy()
tfSH = true_fixed_lighting.cpu().data.numpy()

pSH = open(output_path+'predicted_sh.csv', 'a')
eSH = open(output_path+'expected_sh.csv', 'a')

for i in fSH:
    i.tofile(pSH, sep=',')
    pSH.write('\n')

for i in tfSH:
    i.tofile(eSH, sep=',')
    eSH.write('\n')
'''
print('Generating GIF')
# generate_animation(output_path+'images/', gan_epochs,  exp_name)
