

import os
import h5py
import numpy as np
from utils import *
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Normalize data while loading
class CustomDataSetLoader(Dataset):
    def __init__(self, inData, transform = None):
        self.data = inData
        self.transform = transform
        #normalize here or in __getitem__

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        #data = data.permute(1, 2, 0)
        return data

    def __len__(self):
        return len(self.data)


def get_h5_file_names(path, twoLevel = False):
    h5Files = []

    for file in os.listdir(path):
        h5Files.append(path+file)
    return h5Files

# Following in Synthetic image data loading
# LDAN stores multiple h5 files containing two synthetic image
# and their corresponding SH
# Following function will process thos into one and return
def load_synthetic_ldan_data(path, batch_size = 64):
    h5Files = get_h5_file_names(path)
    if len(h5Files) == 0:
        PRINT('NO H5 FILE FOUND FOR SYNTHETIC IMAGES', 'WARNING')
        return None, None, None

    h5 = h5py.File(h5Files[0], 'r')
    syn1 = h5['data_1']
    syn2 = h5['data_2']
    dlabel = h5['label']
    synIm1 = np.array(syn1[:,:,:]) #np.swapaxes(np.array(dset[:,:,:]), 1, 3)
    synIm2 = np.array(syn2[:,:,:]) #np.swapaxes(np.array(dset2[:,:,:]), 1, 3)
    synLabel = np.array(dlabel)

    totalSets = len(h5Files)
    PRINT(totalSets)
    for i in range(1, totalSets):
        h5 = h5py.File(h5Files[i], 'r')
        syn1 = h5['data_1']
        syn2 = h5['data_2']
        dlabel = h5['label']
        synIm1 = np.concatenate((synIm1, np.array(syn1[:,:,:]))) #np.swapaxes(np.array(dset[:,:,:]), 1, 3)
        synIm2 = np.concatenate((synIm2, np.array(syn2[:,:,:]))) #np.swapaxes(np.array(dset2[:,:,:]), 1, 3)
        synLabel = np.concatenate((synLabel, np.array(dlabel)))
        PRINT(synIm1.shape)
        PRINT(synIm2.shape)
        PRINT(synLabel.shape)

    # Synthetic data was stored from Matlab and hence needs to move the axis
    syn1 = np.moveaxis(synIm1, 1, 3)
    syn1 = np.moveaxis(syn1, 1, 2)
    syn1 = np.moveaxis(syn1, 2, 3)

    syn2 = np.moveaxis(synIm2, 1, 3)
    syn2 = np.moveaxis(syn2, 1, 2)
    syn2 = np.moveaxis(syn2, 2, 3)

    if VERBOSE:
        print('Size of Synthetic Data: ', syn1.shape)
    transform = transforms.Compose([
            #transforms.Resize(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Dataset
    syn1 = CustomDataSetLoader(syn1, transform = transform)
    syn2 = CustomDataSetLoader(syn2, transform = transform)

    # Data Loader for synthetic images
    synImage1 = torch.utils.data.DataLoader(syn1, batch_size = batch_size)
    synImage2 = torch.utils.data.DataLoader(syn2, batch_size = batch_size)
    synLabel = torch.utils.data.DataLoader(synLabel, batch_size = batch_size)

    PRINT('Loading Synthetic Images Completed: ')
    # LDAN works on two frontal pose having same SH: Returning both image set
    return synImage1, synImage2, synLabel

def load_real_images_celebA(path, validation = False, batch_size = 64):
    h5Files = get_h5_file_names(path)
    if len(h5Files) == 0:
        PRINT('NO H5 FILE FOUND FOR SYNTHETIC IMAGES', 'WARNING')
        return None

    # Load data from H5 Files
    firstTime = True
    for file in h5Files:
        hf = h5py.File(file, 'r')
        print hf.keys()
        rImg1 = hf['/Image']
        lighting1 = hf['/Lighting']
        normal1 = hf['/Normal']
        shading1 = hf['/Shading']

        # Following are not need for current experiment
        # PLEASE UNCOMMENT IF YOU NEED
        # height = hf['/Height']
        # reflectance = hf['/Reflectance']
        # finalLoss = hf['/FinalLoss']
        if firstTime:
            rImage = np.array(rImg1[:,:,:])
            lighting = np.array(lighting1[:,:])
            normal = np.array(normal1[:,:,:])
            shading = np.array(shading1[:,:,:])
            firstTime = False
        else:
            rImage = np.concatenate((rImg, np.array(rImg1[:,:,:])))
            lighting = np.concatenate((lighting, np.array(lighting1[:,:])))
            normal = np.concatenate((normal, np.array(normal1[:,:,:])))
            shading = np.concatenate((shading, np.array(shading1[:,:,:])))

    if VERBOSE:
        print('Size of Real data: ', rImage.shape)
    # Transforms being used
    transform = transforms.Compose([
            #transforms.Resize(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    if validation:
        rImage_val, rImage = np.split(rImage, [batch_size])
        lighting_val, lighting = np.split(lighting, [batch_size])
        normal_val, normal = np.split(normal, [batch_size])
        shading_val, shading = np.split(shading, [batch_size])
        
        rImage_val = CustomDataSetLoader(rImage_val, transform = transform)

        real_image_val = torch.utils.data.DataLoader(rImage_val, batch_size= batch_size, shuffle = False)
        sirfs_sh_val = torch.utils.data.DataLoader(lighting_val, batch_size= batch_size, shuffle = False)
        sirfs_normal_val = torch.utils.data.DataLoader(normal_val, batch_size= batch_size, shuffle = False)
        sirfs_shading_val = torch.utils.data.DataLoader(shading_val, batch_size= batch_size, shuffle = False)
        lighting_val = torch.utils.data.DataLoader(lighting_val, batch_size= batch_size, shuffle = False)


    else:
        real_image_val = None
        sirfs_sh_vak = None
        sirfs_normal_val = None
        sirfs_shading_val = None
    # Custom image dataset
    # Normal and Shading is already normalized by SIRFS method
    # So, Normalize only real images
    rImage = CustomDataSetLoader(rImage, transform = transform)

    realImage = torch.utils.data.DataLoader(rImage, batch_size= batch_size, shuffle = False)
    realSH = torch.utils.data.DataLoader(lighting, batch_size= batch_size, shuffle = False)
    rNormal = torch.utils.data.DataLoader(normal, batch_size= batch_size, shuffle = False)
    rShading = torch.utils.data.DataLoader(shading, batch_size= batch_size, shuffle = False)
    
    PRINT('Loading CelebA Real Images Completed')
    # Following are not need for current experiment
    # PLEASE UNCOMMENT IF YOU NEED
    # rHeight = torch.utils.data.DataLoader(height, batch_size= batch_size, shuffle = False)
    # rReflectance = torch.utils.data.DataLoader(reflectance, batch_size= batch_size, shuffle = False)
    # rFinalLoss = torch.utils.data.DataLoader(finalLoss, batch_size= batch_size, shuffle = False)
    return realImage, rNormal, realSH, rShading, real_image_val, sirfs_sh_val, sirfs_normal_val, sirfs_shading_val


   
class DataSetNoPermute(Dataset):
    def __init__(self, inData, transform = None):
        self.data = inData
        self.transform = transform
        #normalize here or in __getitem__

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        data = data.permute(2, 0, 1)
        return data

    def __len__(self):
        return len(self.data)

def get_h5_file_names(path, twoLevel = False):
    h5Files = []
    if twoLevel == True:
        for file in os.listdir(path):
            for f in os.listdir(path+file):
                filePath = path+file+'/'+f
                h5Files.append(filePath)
        return h5Files

    for file in os.listdir(path):
        h5Files.append(path+file)
    return h5Files

def load_SfSNet_data(path, validation = False, twoLevel = False, batch_size = 64):
    h5Files = get_h5_file_names(path, twoLevel)
    if len(h5Files) == 0:
        print('NO H5 FILE FOUND FOR SYNTHETIC IMAGES', 'WARNING')
        return None

    # Load data from H5 Files
    firstTime = True
    for file in h5Files:
        hf = h5py.File(file, 'r')
        print hf.keys()
        rImg1 = hf['/Image']
        lighting1 = hf['/Lighting']
        normal1 = hf['/Normal']
        shading1 = hf['/Shading']
        sirfs_normal1 = hf['/SIRFS_Normal']
        mask1 = hf['/Mask']
        sirfs_lighting1 = hf['/SIRFS_Lighting']
        sirfs_shading1 = hf['/SIRFS_Shading']

        # Following are not need for current experiment
        # PLEASE UNCOMMENT IF YOU NEED
        # height = hf['/Height']
        # reflectance = hf['/Reflectance']
        # finalLoss = hf['/FinalLoss']
        if firstTime:
            rImage = np.array(rImg1[:,:,:])
            lighting = np.array(lighting1[:,:])
            normal = np.array(normal1[:,:,:])
            shading = np.array(shading1[:,:,:])
            sirfs_normal = np.array(sirfs_normal1[:,:,:])
            mask = np.array(mask1[:,:,:])
            sirfs_lighting = np.array(sirfs_lighting1[:,:])
            sirfs_shading = np.array(sirfs_shading1[:,:])
            firstTime = False
        else:
            rImage = np.concatenate((rImage, np.array(rImg1[:,:,:])))
            lighting = np.concatenate((lighting, np.array(lighting1[:,:])))
            normal = np.concatenate((normal, np.array(normal1[:,:,:])))
            shading = np.concatenate((shading, np.array(shading1[:,:,:])))
            sirfs_normal = np.concatenate((sirfs_normal, np.array(sirfs_normal1[:,:,:])))
            mask = np.concatenate((mask, np.array(mask1[:,:,:])))
            sirfs_lighting = np.concatenate((sirfs_lighting, np.array(sirfs_lighting1[:,:])))
            sirfs_shading = np.concatenate((sirfs_shading, np.array(sirfs_shading1[:,:])))


    print('Size of Real data: ', rImage.shape, mask.shape, sirfs_normal.shape)
    # Transforms being used
    transform = transforms.Compose([
            #transforms.Resize(),
            transforms.ToTensor(),
            type(torch.FloatTensor()),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    noNormalize = transforms.Compose([
            #transforms.Resize(),
            transforms.ToTensor(),
    ])
        
    if validation:
        rImage_val, rImage = np.split(rImage, [batch_size])
        lighting_val, lighting = np.split(lighting, [batch_size])
        normal_val, normal = np.split(normal, [batch_size])
        shading_val, shading = np.split(shading, [batch_size])
        mask_val, mask = np.split(mask, [batch_size])
        sirfs_lighting_val, sirfs_lighting = np.split(sirfs_lighting, [batch_size])
        sirfs_normal_val, sirfs_normal = np.split(sirfs_normal, [batch_size])
        sirfs_shading_val, sirfs_shading = np.split(sirfs_shading, [batch_size])

        rImage_val = CustomDataSetLoader(rImage_val, transform = transform)
        #mask_val = CustomDataSetLoader(mask_val, transform = noNormalize)
        #shading_val = CustomDataSetLoader(shading_val, transform = transform)


        real_image_val = torch.utils.data.DataLoader(rImage_val, batch_size= batch_size, shuffle = False)
        lighting_val = torch.utils.data.DataLoader(lighting_val, batch_size= batch_size, shuffle = False)
        shading_val = torch.utils.data.DataLoader(shading_val, batch_size= batch_size, shuffle = False)

        sirfs_sh_val = torch.utils.data.DataLoader(lighting_val, batch_size= batch_size, shuffle = False)
        sirfs_normal_val = torch.utils.data.DataLoader(sirfs_normal_val, batch_size= batch_size, shuffle = False)
        sirfs_shading_val = torch.utils.data.DataLoader(sirfs_shading_val, batch_size= batch_size, shuffle = False)
        normal_val = torch.utils.data.DataLoader(normal_val, batch_size= batch_size, shuffle = False)
        mask_val = torch.utils.data.DataLoader(mask_val, batch_size= batch_size, shuffle = False)
        sirfs_lighting_val = torch.utils.data.DataLoader(sirfs_lighting_val, batch_size= batch_size, shuffle = False)

    else:
        real_image_val = None
        sirfs_sh_val = None
        sirfs_normal_val = None
        sirfs_shading_val = None
        normal_val = None
        mask_val = None
        lighting_val = None
        shading_val = None                              
    # Custom image dataset
    # Normal and Shading is already normalized by SIRFS method
    # So, Normalize only real images
    rImage = CustomDataSetLoader(rImage, transform = transform)
    #mask = CustomDataSetLoader(mask, transform = noNormalize)
    #shading = CustomDataSetLoader(shading, transform = transform)
    #normal = CustomDataSetLoader(normal, transform = noNormalize)

    realImage = torch.utils.data.DataLoader(rImage, batch_size= batch_size, shuffle = False)
    realSH = torch.utils.data.DataLoader(lighting, batch_size= batch_size, shuffle = False)
    rNormal = torch.utils.data.DataLoader(normal, batch_size= batch_size, shuffle = False)
    rShading = torch.utils.data.DataLoader(shading, batch_size= batch_size, shuffle = False)
    sirfs_Normal = torch.utils.data.DataLoader(sirfs_normal, batch_size= batch_size, shuffle = False)
    mask = torch.utils.data.DataLoader(mask, batch_size= batch_size, shuffle = False)
    sirfs_SH = torch.utils.data.DataLoader(sirfs_lighting, batch_size= batch_size, shuffle = False)
    sirfs_shading = torch.utils.data.DataLoader(sirfs_shading, batch_size= batch_size, shuffle = False)
 
    print('Loading SFSNet Synthetic Images Completed')
    # Following are not need for current experiment
    # PLEASE UNCOMMENT IF YOU NEED
    # rHeight = torch.utils.data.DataLoader(height, batch_size= batch_size, shuffle = False)
    # rReflectance = torch.utils.data.DataLoader(reflectance, batch_size= batch_size, shuffle = False)
    # rFinalLoss = torch.utils.data.DataLoader(finalLoss, batch_size= batch_size, shuffle = False)
    return realImage, rNormal, realSH, rShading, mask, sirfs_shading, sirfs_Normal, sirfs_SH, real_image_val, normal_val, lighting_val, shading_val, mask_val, sirfs_shading_val, sirfs_normal_val, sirfs_sh_val  



def getMask(path, batch_size = 64):
    
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder(path, transform)

    mask = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=False)
    
    #real_image_mask_test = next(iter(mask))
    #for real_image_mask_test in mask:
    #    print('Image:')
    #    img, _ = real_image_mask_test
    #    save_image(torchvision.utils.make_grid(img, padding=1), './MASK_TEST.png')

    print('Mask Complte')
    return mask
 
