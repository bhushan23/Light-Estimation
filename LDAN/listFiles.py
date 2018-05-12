import os
path = '/home/bsonawane/Thesis/LightEstimation/SIRFS/realData'
folders = os.listdir(path)
for fold in folders:
    npath = path+'/'+fold
    files = os.listdir(npath)
    for file in files:
        nFile = npath+'/'+file
        print 'FILE:', nFile

