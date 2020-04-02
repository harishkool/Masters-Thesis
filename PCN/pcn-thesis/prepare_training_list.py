import numpy  as np
from open3d import *
from util import read_pcd, save_pcd
import os
import pdb

def prepare_training(pt_dir,fl_name):
    voxl_dir = os.path.join(pt_dir,'voxels')
    catg = os.listdir(voxl_dir)
    fl  = open(fl_name,'w')
    pdb.set_trace()
    for cat in catg:
        cat_dir = os.path.join(voxl_dir,cat)
        voxels = os.listdir(cat_dir)
        total = int(len(voxels)*0.6)
        for voxel in voxels:
            tmpc = voxel.split('_')[0]
            indx = voxel.split('_')[2].split('.')[0]
            trn = tmpc+'_'+indx+'.npz'
            fl.write(trn)
            fl.write('\n')

        # for i in range(0, total):
        #     voxel = voxels[i]
        #     tmpc = voxel.split('_')[0]
        #     indx = voxel.split('_')[2].split('.')[0]
        #     trn = tmpc+'_'+indx+'.npz'
        #     fl.write(trn)
        #     fl.write('\n')
    fl.close()



if __name__=='__main__':
    # prepare_training('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/','train_less_lst.txt')
    prepare_training('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid','valid_lst.txt')