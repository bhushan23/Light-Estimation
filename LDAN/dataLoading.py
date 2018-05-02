from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
from utils import *
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
        data = data.permute(1, 2, 0)
        return data

    def __len__(self):
        return len(self.data)


def get_h5_file_names(path):
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

def load_real_images_celebA(path, batch_size = 64):
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
            rImg = np.array(rImg1[:,:,:])
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
    return realImage, rNormal, realSH, rShading
