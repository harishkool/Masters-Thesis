import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../')
import numpy as np
import pdb
from util import read_pcd, save_pcd
from pyntcloud import PyntCloud
from collections import defaultdict
from open3d import *
import pandas as pd

def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)



if __name__=='__main__':
    # pcd= cat+'_'+str(i)+'.pcd'
    # pcd_pth = os.path.join(cat_dir,pcd)
    pcd_pth = '/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/lamp/lamp_4709.pcd'
    pcd_arr = read_pcd(pcd_pth)
    pdd = pd.DataFrame({"x":pcd_arr[:,0],"y":pcd_arr[:,1],"z":pcd_arr[:,2]})
    cloud = PyntCloud(pdd)
    n_x = 32
    n_y = 32
    n_z = 16
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n_x, n_y=n_y, n_z=n_z)
    voxel_grid = cloud.structures[voxelgrid_id]
    dic = defaultdict(list)
    points = pcd_arr
    voxel_nums = voxel_grid.query(points)
    for indx,voxel_num in enumerate(voxel_nums):
        if not voxel_num in dic:
            dic[voxel_num].append(points[indx])
        else:
            dic[voxel_num].append(points[indx])
    # pdb.set_trace()
    dc_keys = sorted(dic, key=lambda k: len(dic[k]), reverse=True)
    max_points = 40
    max_voxels = 50
    new_lst=[]
    total_voxels = n_x*n_y*n_z
    # voxl_lst = list(dic.keys())
    pdb.set_trace()
    if len(dc_keys) >= 70:
        # vxl_sav = cat+'_vxl'+'_'+str(i)+'.npz'
        # vxl_sav_dir = os.path.join(voxels_cat,vxl_sav)
        vxl_sav_dir = 'sampl_vxl.npz'
        voxl_lst = dc_keys[0:70]
        vxl_nm = 0
        for voxl in voxl_lst:
            points = dic[voxl]
            pnt_len = len(points)
            if pnt_len > max_points:
                vxl_nm += 1
                randomlySelectedY = np.argsort(np.random.random(pnt_len))[:max_points]
                for val in randomlySelectedY:
                    new_lst.append(points[val]) 
                new_lst.append(nw_pnts)
            elif pnt_len < max_points and pnt_len > 6:
                vxl_nm += 1
                zeros = np.zeros((40-pnt_len,3))
                nw_pnts = np.vstack((np.array(points),zeros))
                new_lst.append(nw_pnts)
            elif pnt_len==max_points:
                vxl_nm += 1
                new_lst.append(np.array(points))
    pdb.set_trace()
    print('over')
