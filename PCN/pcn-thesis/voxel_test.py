import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from util import save_pcd, lmdb_dataflow
from collections import defaultdict
import pdb

size = 231792
mx_lst = []
mn_lst = []
txxt = open('lst.txt','w')
max_fl = open('max_lst.txt','w')
min_fl = open('min_lst.txt','w')

df_train, num_train = lmdb_dataflow(
    '/shared/kgcoe-research/mil/harish/pcn_data/data/shapenet/train.lmdb', 1, 2048, 16384, is_training=True)
train_gen = df_train.get_data()
for sz in range(size):
    # pdb.set_trace()
    ids, inputs, gt = next(train_gen)
    pdd = pd.DataFrame({"x":inputs[0][:,0],"y":inputs[0][:,1],"z":inputs[0][:,2]})
    cloud = PyntCloud(pdd)
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
    voxel_grid = cloud.structures[voxelgrid_id]
    dic = defaultdict(list)
    frst = 0
    scnd = 2
    for  i  in range(int(len(inputs[0])/2)):
        points = inputs[0][frst:scnd,:]
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
        txxt.write(str(f))
        txxt.write(' ')
    txxt.write('\n')
    txxt.write('\n')
    max_points = max(lsst)
    mx_lst.append(max_points)
    min_points = min(lsst)
    mn_lst.append(min_points)
txxt.close()
# pdb.set_trace()
for val in mx_lst:
    max_fl.write(str(val))
    max_fl.write('\n')
max_fl.close()
for val in mn_lst:
    min_fl.write(str(val))
    min_fl.write('\n')
min_fl.close()
print('Max list: {}'.format(mx_lst))
print('Min list: {}'.format(mn_lst))
