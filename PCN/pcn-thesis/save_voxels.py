import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../')
import numpy as np
# from point_cloud_ops import points_to_voxel
import pandas as pd
import pdb
from util import read_pcd, save_pcd
from pyntcloud import PyntCloud
from collections import defaultdict
from open3d import *


'''
sofa     -  23364
cabinet   - 10568
car       - 45446
table     - 45924
airplane  - 30320
chair      - 46010
lamp      -  16260
watercraft - 13520
'''

'''
16 x 16 x 16
sofa        - 21802
cabineet    -  8888
car         -  42098
table       -  41153
airplane    -  29473
chair       -  41435
lamp        -  9473
watercraft  -  12047
'''
#and cat in ['airplane', 'watercraft', 'lamp']
#'lamp_4709.npz'
#'table_6678.npz'
def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)

def voxel_gen(pth_dir):
    catg = os.listdir(pth_dir)
    n_x = 16
    n_y = 16
    n_z = 16
    voxel_dir = pth_dir + 'voxels'
    if not os.path.isdir(voxel_dir):
        os.mkdir(voxel_dir)
    # pdb.set_trace()
    for cat in catg:
        if not cat=='valid' and not cat in ['airplane', 'watercraft', 'lamp', 'voxels']:
            cat_dir = os.path.join(pth_dir,cat)
            all_pcds = os.listdir(cat_dir)
            voxels_cat = os.path.join(voxel_dir, cat)
            if not os.path.isdir(voxels_cat):
                os.mkdir(voxels_cat)
            for i in range(1, int(len(all_pcds)/2)+1):
                pcd= cat+'_'+str(i)+'.pcd'
                pcd_pth = os.path.join(cat_dir,pcd)
                pcd_arr = read_pcd(pcd_pth)
                pdd = pd.DataFrame({"x":pcd_arr[:,0],"y":pcd_arr[:,1],"z":pcd_arr[:,2]})
                cloud = PyntCloud(pdd)
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
                max_voxels = 70
                new_lst=[]
                total_voxels = n_x*n_y*n_z
                # voxl_lst = list(dic.keys())
                if len(dc_keys) >= max_voxels:
                    vxl_sav = cat+'_vxl'+'_'+str(i)+'.npz'
                    vxl_sav_dir = os.path.join(voxels_cat,vxl_sav)
                    voxl_lst = dc_keys[0:max_voxels]
                    vxl_nm = 0
                    for voxl in voxl_lst:
                        points = dic[voxl]
                        pnt_len = len(points)
                        if pnt_len > max_points:
                            vxl_nm += 1
                            randomlySelectedY = np.argsort(np.random.random(pnt_len))[:max_points]
                            for val in randomlySelectedY:
                                new_lst.append(points[val]) 
                            # new_lst.append(nw_pnts)
                        elif pnt_len < max_points and pnt_len > 6:
                            vxl_nm += 1
                            zeros = np.zeros((40-pnt_len,3))
                            nw_pnts = np.vstack((np.array(points),zeros))
                            new_lst.append(nw_pnts)
                        elif pnt_len==max_points:
                            vxl_nm += 1
                            new_lst.append(np.array(points))
                    # pdb.set_trace()
                    voxel_arry = np.array(new_lst)
                    shpe = voxel_arry.shape
                    if shpe == (max_voxels,max_points,3):
                        np.savez(vxl_sav_dir,voxels=voxel_arry)



if __name__ == "__main__":
    voxel_gen('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid/')
    # pdb.set_trace()
    # print(voxel_stack.shape)



## 32 x 32 x 32 --> airplane , sofa, watercraft

#lamp --> pcd_text_new --> 32 x 32 x 16