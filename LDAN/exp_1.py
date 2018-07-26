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
load_syn = True
load_GAN = True
train_syn = False
train_GAN = True
FirstRun = False #True
gan_epochs = 50
syn_epochs = 150
exp_name = 'resnet_CelebA'
LOCAL_MACHINE = False
output_path = './resnet_CelebA/'

synthetic_data_path =  '/home/bsonawane/Thesis/LightEstimation/SIRFS/synImages/'   #'/home/bsonawane/Thesis/LightEstimation/Light-Estimation/data/AE/syn/'  #'/home/bhushan/college/CV/Thesis/Projects/lightestimation/Light-Estimation/data/AE/syn/'
real_data_path      = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData/data/' 
#'/home/bsonawane/Thesis/Estimation/Light-Estimation/data/AE/real/'   #'/home/bhushan/college/CV/Thesis/Projects/lightestimation/Light-Estimation/data/AE/real/'

real_val_data_path = '/home/bsonawane/Thesis/LightEstimation/SIRFS/celebA/valData/'

if FirstRun == True:
    os.mkdir(output_path)
    os.mkdir(output_path+'images/')
    os.mkdir(output_path+'val/')
    os.mkdir(output_path+'models/')
    os.mkdir(output_path+'savedModels/')


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
    if Predicted == True:
        outShadingB = denorm(outShadingB)
    outShadingB = applyMask(outShadingB, real_image_mask)
    outShadingB = outShadingB.data
    #pic = torchvision.utils.make_grid(outShadingB, padding=1)
    save_image(outShadingB, path + name+'.png')
    save_image(outShadingB[0], path + name+'_0.png')
    return outShadingB

def to_numpy(v):
    return v.cpu().data.numpy()

def save_results_h5(sirfs_fixed_normal, tfSH, fSH, mask, output_path):
    hf = h5py.File(output_path+'results.h5', 'w')
    s_f_normal = sirfs_fixed_normal #denorm(sirfs_fixed_normal)
    t_f_normal = applyMask(t_f_normal, mask)
    #s_f_normal = applyMask(s_f_normal, mask)

    hf.create_dataset('sirfs_normal', data = to_numpy(s_f_normal))
    hf.create_dataset('expected_sh', data = tfSH)
    hf.create_dataset('predicted_sh', data = fSH)
    hf.create_dataset('mask', data = to_numpy(var(mask)))
    hf.close()

def var(x):
    if IS_CUDA:
        x = x.cuda()
    return Variable(x)
# End of Helper routines

syn_image, syn_normal, syn_sh, syn_shading, syn_mask, syn_sirfs_shading, syn_sirfs_normal, syn_sirfs_sh, syn_image_val, syn_normal_val, syn_lighting_val, syn_shading_val, syn_mask_val, syn_sirfs_shading_val, syn_sirfs_normal_val, syn_sirfs_sh_val  = dataLoading.load_SfSNet_data(synthetic_data_path, twoLevel = True)

#real_image, real_normal, real_sh, real_shading, real_mask, real_sirfs_shading, real_sirfs_normal, real_sirfs_sh, real_image_val, real_normal_val, real_lighting_val, real_shading_val, real_mask_val, real_sirfs_shading_val, real_sirfs_normal_val, real_sirfs_sh_val  = dataLoading.load_SfSNet_data(real_data_path, validation = True, twoLevel = True)


real_image, real_sirfs_normal, real_sirfs_sh, real_sirfs_shading, real_mask, real_image_val, real_sirfs_sh_val, real_sirfs_normal_val, real_sirfs_shading_val, real_mask_val = dataLoading.load_real_images_celebA(real_data_path)


real_image_val, real_sirfs_normal_val, real_sirfs_sh_val, real_sirfs_shading_val, real_mask_val, _, _, _, _, _ = dataLoading.load_real_images_celebA(real_val_data_path, load_mask = True)

real_image_mask_test = next(iter(real_mask_val))
utils.save_image(torchvision.utils.make_grid(real_image_mask_test*255, padding=1), output_path+'images/MASK_TEST.png')

tmp = var(next(iter(real_image_val)))
tmp = denorm(tmp)
print(tmp.data.shape)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_syn_image.png')

tmp = var(next(iter(real_sirfs_normal_val)))
tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_syn_normal.png')

tmp = var(next(iter(real_sirfs_shading_val)))
#tmp = denorm(tmp)
tmp = applyMask(tmp, real_image_mask_test)
utils.save_image(torchvision.utils.make_grid(tmp.data, padding=1), output_path+'images/test_sirf_shading.png')

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
    syn_net_train(featureNet, lightingNet, syn_image, syn_image, syn_sh, num_epochs = syn_epochs)
    # save_image(predict(featureNet, lightingNet, synVal1), outPath+'_Synthetic_Image.png')
    torch.save(featureNet.state_dict(), output_path+'models/featureNet.pkl')
    torch.save(lightingNet.state_dict(),output_path+ 'models/lightingNet.pkl')


fixed_input = var(next(iter(real_image_val))).type(dtype)
sirfs_fixed_normal = var(next(iter(real_sirfs_normal_val)))
fixed_mask =  real_image_mask_test #next(iter(real_mask_val))
sirfs_fixed_sh = var(next(iter(real_sirfs_sh_val)))
  
if load_GAN:
    lightingNet.load_state_dict(torch.load(output_path+'models/GAN_LNet.pkl'))
    R.load_state_dict(torch.load(output_path+'models/Generator.pkl'))


if train_GAN:
    if real_image_val == None:
        real_image_val = real_image
        sirfs_normal_val = sirfs_SH

    trainGAN(lightingNet, R, D, featureNet, syn_image, real_image, real_sirfs_sh, fixed_input, sirfs_fixed_normal, real_image_mask_test, sirfs_fixed_sh, output_path = output_path, num_epoch = gan_epochs)
    torch.save(lightingNet.state_dict(), output_path+'models/GAN_LNet.pkl')
    torch.save(R.state_dict(),output_path+ 'models/Generator.pkl')


## TESTING 
fixedSH = lightingNet(R(fixed_input))
fixedSH = fixedSH.type(torch.DoubleTensor)
# print('OUTPUT OF fixedSH:', fixedSH.data.size(), sirfs_fixed_normal.size())

syn_sh_out = lightingNet(featureNet(fixed_input))
syn_sh_out = syn_sh_out.type(torch.DoubleTensor)


#print(sirfs_fixed_normal)
#print(true_fixed_normal)
## With SIRFS_NORMAL
save_shading(sirfs_fixed_normal, syn_sh_out, real_image_mask_test, path = output_path+'val/', name = 'PREDICTED_SYN_SIRFS_NORMAL', shadingFromNet = True, Predicted = True)



#print(sirfs_fixed_normal)
#print(true_fixed_normal)
## With SIRFS_NORMAL
save_shading(sirfs_fixed_normal, fixedSH, real_image_mask_test, path = output_path+'val/', name = 'PREDICTED_SIRFS_NORMAL', shadingFromNet = True, Predicted = True)

## EXPECTED with SIRFS NORMAL
#save_shading(sirfs_fixed_normal, true_fixed_lighting, real_image_mask_test, path = output_path+'val/', name = 'EXPECTED_SIRFS_NORMAL', shadingFromNet = True)
'''

fSH = fixedSH.cpu().data.numpy()
tfSH = sirfs_fixed_sh.cpu().data.numpy()

save_results_h5(sirfs_fixed_normal, tfSH, fSH, real_image_mask_test, output_path)

pSH = open(output_path+'predicted_sh.csv', 'a')
eSH = open(output_path+'expected_sh.csv', 'a')

for i in fSH:
    i.tofile(pSH, sep=',')
    pSH.write('\n')

for i in tfSH:
    i.tofile(eSH, sep=',')
    eSH.write('\n')

print('Generating GIF')
#generate_animation(output_path+'images/', gan_epochs,  exp_name)

'''

'''
# Testing
if FirstRun == False:
if SHOW_IMAGES:
    dreal = next(iter(realImage))
    show(dreal[0])
    dNormal = next(iter(rNormal))
    show(denorm(dNormal[0]))
'''
'''
lightingNet = lightingNet.cpu()
D = D.cpu()
R = R.cpu()
torch.save(lightingNet.state_dict(), './CPU_GAN_LNet.pkl')
torch.save(D.state_dict(), './CPU_Discriminator.pkl')
torch.save(R.state_dict(), './CPU_Generator.pkl')
'''
