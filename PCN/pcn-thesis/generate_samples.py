import tensorflow as tf
import os
import sys
sys.path.append('sampling')
# from tf_sampling import farthest_point_sample, gather_point
import tf_sampling
import numpy as np
import glob
import pdb

def generate_samples(pcd_dir, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    cat = os.listdir(pcd_dir)
    pdb.set_trace()
    for dir in cat: 
        cat_dir = os.path.join(pcd_dir,cat)
        all_pcd = glog.glob(cat_dir+'*.pcd')




if __name__=='__main__':
    generate_samples('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/','/shared/kgcoe-research/mil/harish/pcn_data/sampled_data/')