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


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)

def voxel_gen(pth_dir):
    catg = os.listdir(pth_dir)
    n_x = 32
    n_y = 16
    n_z = 16
    pdb.set_trace()
    for cat in catg:
        if cat=='lamp':
           cat_dir = os.path.join(pth_dir,cat)
           fl = open(os.path.join(pth_dir, cat+'.txt'),'w')
           all_pcds = os.listdir(cat_dir)
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
                # max_points = max([len(val) for val in dic.values()])

                max_points = 40
                new_lst=[]
                pdb.set_trace()
                total_voxels = n_x*n_y*n_z
                voxl_lst = list(dic.keys())
                vxl_nm = 0
                for voxl in voxl_lst:
                    points = dic[voxl]
                    pnt_len = len(points)
                    if pnt_len > max_points:
                        vxl_nm += 1
                        randomlySelectedY = np.argsort(np.random.random(pnt_len))[:max_points]
                        for val in randomlySelectedY:
                            new_lst.append(points[val]) 
                        # nw_pnts = points[randomlySelectedY]
                        new_lst.append(nw_pnts)
                    elif pnt_len < max_points and pnt_len > 6:
                        vxl_nm += 1
                        zeros = np.zeros((40-pnt_len,3))
                        nw_pnts = np.vstack((np.array(points),zeros))
                        new_lst.append(np.array(points))
                    elif pnt_len==max_points:
                        vxl_nm += 1
                        new_lst.append(np.array(points))
                new_lst = np.array(new_lst)
                fl.write(str(vxl_nm))
                fl.write('\n')
           fl.close()



if __name__ == "__main__":
    voxel_gen('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/')
    # pdb.set_trace()
    # print(voxel_stack.shape)



## 32 x 32 x 32 --> airplane , sofa, watercraft

#lamp --> pcd_text_new --> 32 x 32 x 16