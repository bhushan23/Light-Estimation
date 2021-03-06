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
load_syn = False #True
load_GAN = False #True
train_syn = True #False
train_GAN = True #False
FirstRun = False
gan_epochs = 500
syn_epochs = 100
exp_name = 'resnet_SfSNet'
LOCAL_MACHINE = False
output_path = './resnet_SfSNet/'
synthetic_image_dataset_path = './data/synHao/'
sfs_net_path = '/home/bsonawane/Thesis/LightEstimation/SIRFS/synImages/'   #scripts/SfsNet_SynImage_back/'
if LOCAL_MACHINE:
    real_image_dataset_path = '../../Light-Estimation/datasets/realImagesSH/'
else:
    real_image_dataset_path = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData/data/'


if FirstRun == True:
    os.mkdir(output_path)
    os.mkdir(output_path+'images/')
    os.mkdir(output_path+'val/')
    os.mkdir(output_path+'models/')
    os.mkdir(output_path+'savedModels/')


real_image_mask = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData/mask/'
global_batch_size = 64

#if not os.path.exists(output_image_path):
#    os.makedirs(output_image_path)

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
syn_image1, syn_image2, syn_label = dataLoading.load_synthetic_ldan_data(synthetic_image_dataset_path)
#real_image, sirfs_normal, sirfs_SH, sirfs_shading, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val = dataLoading.load_real_images_celebA(real_image_dataset_path, validation = True)
#real_image_mask = dataLoading.getMask(real_image_mask, global_batch_size)
#real_image, sirfs_normal, sirfs_SH, sirfs_shading, tNormal, real_image_mask, tSH, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val, true_normal_val, mask_val, true_lighting_val = dataLoading.load_SfSNet_data(sfs_net_path, validation = True, twoLevel = True)


real_image, real_normal, real_sh, real_shading, mask, sirfs_shading, sirfs_normal, sirfs_sh, real_image_val, real_normal_val, real_lighting_val, real_shading_val, mask_val, sirfs_shading_val, sirfs_normal_val, sirfs_sh_val  = dataLoading.load_SfSNet_data(sfs_net_path, validation = True, twoLevel = True)


# Transforms being used
#if SHOW_IMAGES:

real_image_mask_test = next(iter(mask_val))
utils.save_image(torchvision.utils.make_grid(real_image_mask_test*255, padding=1), output_path+'images/MASK_TEST.png')

tmp = next(iter(syn_image1))
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/test_synthetic_img.png')

tmp = var(next(iter(real_image_val)))
tmp = denorm(tmp)
print(tmp.data.shape)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_real_image.png')

tmp = var(next(iter(real_normal_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_real_normal.png')


tmp = var(next(iter(sirfs_normal_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_sirf_normal.png')

tmp = var(next(iter(sirfs_shading_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_sirf_shading.png')


tmp = var(next(iter(real_shading_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data*255, padding=1), output_path+'images/test_real_shading.png')


## TRUE SHADING

'''
tmp = next(iter(sirfs_shading_val))
tmp = utils.denorm(tmp)
tmp = applyMask(var(tmp).type(torch.DoubleTensor), real_image_mask_test) 
tmp = tmp.data
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/Validation_SIRFS_SHADING.png')
'''

featureNet = models.ResNet(models.BasicBlock, [2, 2, 2, 2], 27)
#featureNet = models.BaseSimpleFeatureNet()
lightingNet = models.LightingNet()
D = models.Discriminator()
R = models.ResNet(models.BasicBlock, [2, 2, 2, 2], 27) #
#R = models.BaseSimpleFeatureNet()

print(featureNet)
print(lightingNet)
featureNet = featureNet.cuda()
lightingNet = lightingNet.cuda()
D = D.cuda()
R = R.cuda()

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
# Training
if load_syn:
    featureNet.load_state_dict(torch.load(output_path+'models/featureNet.pkl'))
    lightingNet.load_state_dict(torch.load(output_path+ 'models/lightingNet.pkl'))

if train_syn:
    syn_net_train(featureNet, lightingNet, syn_image1, syn_image2, syn_label, num_epochs = syn_epochs)
    # save_image(predict(featureNet, lightingNet, synVal1), outPath+'_Synthetic_Image.png')
    torch.save(featureNet.state_dict(), output_path+'models/featureNet.pkl')
    torch.save(lightingNet.state_dict(),output_path+ 'models/lightingNet.pkl')


fixed_input = var(next(iter(real_image_val))).type(dtype)
sirfs_fixed_normal = var(next(iter(sirfs_normal_val)))
true_fixed_normal = var(next(iter(real_normal_val)))
true_fixed_lighting = var(next(iter(real_lighting_val)))

#real_image_mask = next(iter(real_image_mask))
#utils.show(torchvision.utils.make_grid(utils.denorm(fixed_input), padding=1))
   
if load_GAN:
    lightingNet.load_state_dict(torch.load(output_path+'models/GAN_LNet.pkl'))
    R.load_state_dict(torch.load(output_path+'models/Generator.pkl'))


if train_GAN:
    fs = predictAllSynthetic(featureNet, syn_image1)
    if real_image_val == None:
        real_image_val = real_image
        sirfs_normal_val = sirfs_SH

    trainGAN(lightingNet, R, D, fs, real_image, sirfs_sh, fixed_input, sirfs_fixed_normal, real_image_mask_test, output_path = output_path, num_epoch = gan_epochs)
    torch.save(lightingNet.state_dict(), output_path+'models/GAN_LNet.pkl')
    torch.save(R.state_dict(),output_path+ 'models/Generator.pkl')


## TESTING 
fixedSH = lightingNet(R(fixed_input))
fixedSH = fixedSH.type(torch.DoubleTensor)
# print('OUTPUT OF fixedSH:', fixedSH.data.size(), sirfs_fixed_normal.size())
#print('expected SH:', true_fixed_lighting)
#print('predicted SH:', fixedSH)

## With SIRFS_NORMAL
save_shading(sirfs_fixed_normal, fixedSH, real_image_mask_test, path = output_path+'val/', name = 'PREDICTED_SIRFS_NORMAL', shadingFromNet = True, Predicted = True)

## With True Normal
save_shading(true_fixed_normal, fixedSH, real_image_mask_test, path = output_path+'val/', name = 'PREDICTED_TRUE_NORMAL', shadingFromNet = True, Predicted = True)

## EXPECTED with SIRFS NORMAL
save_shading(sirfs_fixed_normal, true_fixed_lighting, real_image_mask_test, path = output_path+'val/', name = 'EXPECTED_SIRFS_NORMAL', shadingFromNet = True)

## EXPECTED with true normal
save_shading(true_fixed_normal, true_fixed_lighting, real_image_mask_test, path = output_path+'val/', name = 'EXPECTED_TRUE_NORMAL', shadingFromNet = True)


## Save single image
#tmp = next(iter(sirfs_shading))
#utils.save_image(tmp[0], output_path+'val/shading.png')

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

print('Generating GIF')
generate_animation(output_path+'images/', gan_epochs,  exp_name)



'''
# Testing
if FirstRun == False:
if SHOW_IMAGES:
    dreal = next(iter(realImage))
    show(dreal[0])
    dNormal = next(iter(rNormal))
    show(denorm(dNormal[0]))
'''

lightingNet = lightingNet.cpu()
D = D.cpu()
R = R.cpu()
torch.save(lightingNet.state_dict(), './CPU_GAN_LNet.pkl')
torch.save(D.state_dict(), './CPU_Discriminator.pkl')
torch.save(R.state_dict(), './CPU_Generator.pkl')
