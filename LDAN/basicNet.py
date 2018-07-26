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
import sys
sys.path.insert(0, './core')

# Custom import
import dataLoading
import utils
from lossfunctions import *
from shading import *
import models
from utils import PRINT
from train import *

SHOW_IMAGES = False
TrainSyn = True
TrainGAN = True
FirstRun = False
LOCAL_MACHINE = False
output_path = './BasicNet/'
synthetic_image_dataset_path = './data/synHao_T/'

if LOCAL_MACHINE:
    real_image_dataset_path = '../../Light-Estimation/datasets/realImagesSH/'
else:
    real_image_dataset_path = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData/data_T/'

real_image_mask = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData/mask/'
global_batch_size = 64

#if not os.path.exists(output_image_path):
#    os.makedirs(output_image_path)

# Helper routines
IS_CUDA = False
if torch.cuda.is_available():
    IS_CUDA = True

def var(x):
    if IS_CUDA:
        x = x.cuda()
    return Variable(x)
# End of Helper routines

# Load synthetic dataset
syn_image1, syn_image2, syn_label = dataLoading.load_synthetic_ldan_data(synthetic_image_dataset_path)
real_image, sirfs_normal, sirfs_SH, sirfs_shading, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val = dataLoading.load_real_images_celebA(real_image_dataset_path, validation = True)
real_image_mask = dataLoading.getMask(real_image_mask, global_batch_size)

# Transforms being used
#if SHOW_IMAGES:
tmp = next(iter(syn_image1))
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/test_synthetic_img.png')
tmp = next(iter(real_image))
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/test_real_image.png')
tmp = next(iter(sirfs_normal))
utils.save_image(torchvision.utils.make_grid(utils.denorm(tmp), padding=1), output_path+'images/test_sirf_normal.png')
real_image_mask_test, _ = next(iter(real_image_mask))
utils.save_image(torchvision.utils.make_grid(real_image_mask_test, padding=1), output_path+'images/MASK_TEST.png')
tmp = next(iter(sirfs_shading_val))
tmp = utils.denorm(tmp)
tmp = applyMask(var(tmp).type(torch.DoubleTensor), real_image_mask_test) 
tmp = tmp.data
utils.save_image(torchvision.utils.make_grid(tmp, padding=1), output_path+'images/Validation_SIRFS_SHADING.png')


# featureNet = ResNet(BasicBlock, [2, 2, 2, 2], 27)
featureNet = models.BaseSimpleFeatureNet()
lightingNet = models.LightingNet()
D = models.Discriminator()
# R = models.ResNet(models.BasicBlock, [2, 2, 2, 2], 27) #
R = models.BaseSimpleFeatureNet()

print(featureNet)
print(lightingNet)
featureNet = featureNet.cuda()
lightingNet = lightingNet.cuda()
D = D.cuda()
R = R.cuda()

dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!
# Training
if TrainSyn:
    syn_net_train(featureNet, lightingNet, syn_image1, syn_image2, syn_label, num_epochs = 200)
    # save_image(predict(featureNet, lightingNet, synVal1), outPath+'_Synthetic_Image.png')
    torch.save(featureNet.state_dict(), output_path+'models/featureNet.pkl')
    torch.save(lightingNet.state_dict(),output_path+ 'models/lightingNet.pkl')
else:
    featureNet.load_state_dict(torch.load(output_path+'models/featureNet.pkl'))
    lightingNet.load_state_dict(torch.load(output_path+ 'models/lightingNet.pkl'))

fixed_input = var(next(iter(real_image_val))).type(dtype)
sirfs_fixed_normal = var(next(iter(sirfs_normal_val)))
#real_image_mask = next(iter(real_image_mask))
#utils.show(torchvision.utils.make_grid(utils.denorm(fixed_input), padding=1))
   

if TrainGAN:
    fs = predictAllSynthetic(featureNet, syn_image1)
    if real_image_val == None:
        real_image_val = real_image
        sirfs_normal_val = sirfs_SH

    trainGAN(lightingNet, R, D, fs, real_image, sirfs_SH, fixed_input, sirfs_fixed_normal, real_image_mask_test, output_path = output_path, num_epoch = 300)
else:    
    lightingNet.load_state_dict(torch.load(output_path+'models/GAN_LNet.pkl'))
    R.load_state_dict(torch.load(output_path+'models/Generator.pkl'))

'''
# Testing
if FirstRun == False:
if SHOW_IMAGES:
    dreal = next(iter(realImage))
    show(dreal[0])
    dNormal = next(iter(rNormal))
    show(denorm(dNormal[0]))


lightingNet = lightingNet.cpu()
D = D.cpu()
R = R.cpu()
torch.save(lightingNet.state_dict(), './GAN_LNet.pkl')
torch.save(D.state_dict(), './Discriminator.pkl')
torch.save(R.state_dict(), './Generator.pkl')
'''
