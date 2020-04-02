import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from util import save_pcd, lmdb_dataflow
from collections import defaultdict
import pdb
from open3d import *
import os

dir_name = '/shared/kgcoe-research/mil/harish/pcn_data/pcd_data/'
txt_dir = '/shared/kgcoe-research/mil/harish/pcn_data/pcd_text_new2/'
dir_lst = os.listdir(dir_name)
for indx, dir in enumerate(dir_lst):
    nw_path = os.path.join(dir)
    all_pcd = os.listdir(os.path.join(dir_name, nw_path))
    # print(all_pcd)
    cat_dir = os.path.join(txt_dir,dir_lst[indx])
    if not os.path.isdir(cat_dir):
        os.mkdir(cat_dir)
    tmp = os.path.join(cat_dir,dir_lst[indx])
    # pdb.set_trace()
    cat_txt = open(tmp+'.txt','w')
    mx_txt = open(tmp+'_max.txt','w')
    mn_txt = open(tmp+'_min.txt','w')
    mx_lst = []
    mn_lst = []
    for pcd_file in all_pcd:
        # pdb.set_trace()
        pcd_fl = os.path.join(dir_name, nw_path, pcd_file)
        pcd = read_point_cloud(pcd_fl)
        pcd_array = np.array(pcd.points)
        pdd = pd.DataFrame({"x":pcd_array[:,0],"y":pcd_array[:,1],"z":pcd_array[:,2]})
        cloud = PyntCloud(pdd)
        voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=16, n_z=16)
        voxel_grid = cloud.structures[voxelgrid_id]
        dic = defaultdict(list)
        frst = 0
        scnd = 2
        for  i  in range(int(len(pcd_array)/2)):
            points = pcd_array[frst:scnd,:]
            frst+=2
            scnd+=2
            voxel_nums = voxel_grid.query(points)
            for indx,voxel_num in enumerate(voxel_nums):
                if not voxel_num in dic:
                    dic[voxel_num].append(points[indx])
                else:
                    dic[voxel_num].append(points[indx])
        lsst = [len(val) for val in dic.values()]
        for f in lsst:
            cat_txt.write(str(f))
            cat_txt.write(' ')
        cat_txt.write('\n')
        max_points = max(lsst)
        mx_lst.append(max_points)
        min_points = min(lsst)
        mn_lst.append(min_points)
    cat_txt.close()
    # pdb.set_trace()
    for val in mx_lst:
        mx_txt.write(str(val))
        mx_txt.write('\n')
    mx_txt.close()
    for val in mn_lst:
        mn_txt.write(str(val))
        mn_txt.write('\n')
    mn_txt.close()
    # print('Max list: {}'.format(mx_lst))
    # print('Min list: {}'.format(mn_lst))
