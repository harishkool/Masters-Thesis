import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
from util import save_pcd, lmdb_dataflow
from collections import defaultdict
import pdb
import os
from util import save_pcd, lmdb_dataflow

size = 231792
mx_lst = []
mn_lst = []
dc = {'03001627':'chair','02691156':'airplane','02958343':'car','04256520':'sofa','03636649':'lamp',
        '04379243':'table','02933112':'cabinet','04530566':'watercraft'}
df_train, num_train = lmdb_dataflow(
    '/shared/kgcoe-research/mil/harish/pcn_data/data/shapenet/valid.lmdb', 1, 2048, 16384, is_training=False)
shared_path = '/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid'
train_gen = df_train.get_data()
cnt = 1
prefix = None
dcc = {}
if not os.path.isdir(shared_path):
    os.mkdir(shared_path)
# size = df_train.size()
size = num_train
# pdb.set_trace()
for sz in range(size):
    # pdb.set_trace()
    ids, inputs, gt = next(train_gen)
    cat = dc[str(ids[0].split('_')[0])]
    dir_path = os.path.join(shared_path,cat)
    if cat not in dcc:
        dcc[cat]=1
    else:
        dcc[cat] += 1

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    pc = inputs[0]
    pcd = np.copy(pc)
    gt_pc = gt[0]
    gt_pcd = np.copy(gt_pc)
    fl_name = os.path.join(dir_path, cat+'_'+str(dcc[cat]))
    save_pcd(fl_name+'.pcd', pcd)
    save_pcd(fl_name+'_gt.pcd', gt_pcd)
    