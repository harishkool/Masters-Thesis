from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
from scipy.io import loadmat
import pdb

class PhenixDataset(data.Dataset):
    def __init__(self,
                 split='train',
                 dataset_path= None):
        
        self.dataset = dataset_path
        if split == 'train':
            self.dataset = os.path.join(self.dataset, 'train')
        else:
            self.dataset = os.path.join(self.dataset, 'test')
        self.dir_list = os.listdir(self.dataset)
        self.all_data = []
        for dr in self.dir_list:
            mat_files = []
            fc_fl = os.path.join(self.dataset, dr, 'faces.mat')
            vc_fl = os.path.join(self.dataset, dr, 'vertices.mat')
            fcvx_fl = os.path.join(self.dataset, dr, 'facevertxdata.mat')
            mat_files.append(fc_fl)
            mat_files.append(vc_fl)
            mat_files.append(fcvx_fl)
            self.all_data.append(mat_files)


    def __getitem__(self, index):
        # print(self.dir_list[index])
        mat_files = self.all_data[index]
        fc_fl = loadmat(mat_files[0])
        vc_fl = loadmat(mat_files[1])
        fc_vx = loadmat(mat_files[2])
        faces = fc_fl['fces']
        vertices = vc_fl['vrtcs']
        all_labels = fc_vx['fcverxt_data']
        inputs =[]
        gt=[]
        for indx, face in enumerate(faces):
            vrtx1 = list(vertices[face[0]])
            vrtx2 = list(vertices[face[1]])
            vrtx3 = list(vertices[face[2]])
            features = vrtx1 + vrtx2 + vrtx3
            inputs.append(np.array([features]))
            # print(all_labels[indx][0],all_labels[indx][1])
            if all_labels[indx][0]==1:
                gt.append(0)
            elif all_labels[indx][1]==1 or all_labels[indx][2]==1:
                gt.append(1)
        inputs = np.array(inputs)
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[2])
        gt= np.array(gt)
        # np.save(self.dir_list[index]+'.npz',labels=gt)
        return inputs, gt

    def __len__(self):
        return len(self.all_data)

if __name__ == '__main__':
    # datapath = sys.argv[1]
    dataset = PhenixDataset(dataset_path='/home/nagaharish/Downloads/pointnet.pytorch-master/data_mesh/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    # for i in range(len(dataset)):
    #     inputs, gt = dataset[i]
    #     print(inputs.shape)
    #     print(gt.shape)
    for i, data in enumerate(dataloader, 0):
        inputs, gt = data
        print(inputs.shape)
        print(gt.shape)



