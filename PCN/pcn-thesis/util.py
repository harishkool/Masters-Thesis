import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import matplotlib.pyplot as plt
import numpy as np
from open3d import *
import numpy as np
from tensorpack import dataflow
import lmdb
import pdb

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class PreprocessData(dataflow.ProxyDataFlow):
    def __init__(self, ds, input_size, output_size):
        super(PreprocessData, self).__init__(ds)
        self.input_size = input_size
        self.output_size = output_size

    def get_data(self):
        for id, input, gt in self.ds.get_data():
            input = resample_pcd(input, self.input_size)
            gt = resample_pcd(gt, self.output_size)
            yield id, input, gt


class BatchData(dataflow.ProxyDataFlow):
    def __init__(self, ds, batch_size, input_size, gt_size, remainder=False, use_list=False):
        super(BatchData, self).__init__(ds)
        self.batch_size = batch_size
        self.input_size = input_size
        self.gt_size = gt_size
        self.remainder = remainder
        self.use_list = use_list

    def __len__(self):
        ds_size = len(self.ds)
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self):
        print('called iter method atrtibute')
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield self._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield self._aggregate_batch(holder, self.use_list)

    def _aggregate_batch(self, data_holder, use_list=False):
        ''' Concatenate input points along the 0-th dimension
            Stack all other data along the 0-th dimension
        '''
        # pdb.set_trace()
        ids = np.stack([x[0] for x in data_holder])
        # print('x0 shape {}'.format(data_holder[0][0].shape))
        print('x1 shape {}'.format(data_holder[0][1].shape))
        inputs = [resample_pcd(x[1], self.input_size) if x[1].shape[0] > self.input_size else x[1]
            for x in data_holder]
        inputs = np.expand_dims(np.concatenate([x for x in inputs]), 0).astype(np.float32)
        npts = np.stack([x[1].shape[0] if x[1].shape[0] < self.input_size else self.input_size
            for x in data_holder]).astype(np.int32)
        gts = np.stack([resample_pcd(x[2], self.gt_size) for x in data_holder]).astype(np.float32)
        return ids, inputs, npts, gts


# def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
#     df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
#     size = df.size()
#     print('size of train lmdb is {}'.format(size))
#     if is_training:
#         df = dataflow.LocallyShuffleData(df, buffer_size=2000)
#         df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
#     df = BatchData(df, batch_size, input_size, output_size)
#     if is_training:
#         df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
#     df = dataflow.RepeatedData(df, -1)
#     if test_speed:
#         dataflow.TestDataSpeed(df, size=1000).start()
#     df.reset_state()
#     return df, size


def lmdb_dataflow(lmdb_path, batch_size, input_size, output_size, is_training, test_speed=False):
    df = dataflow.LMDBSerializer.load(lmdb_path, shuffle=False)
    size = df.size()
    if is_training:
        df = dataflow.LocallyShuffleData(df, buffer_size=2000)
    df = dataflow.PrefetchData(df, nr_prefetch=500, nr_proc=1)
    df = PreprocessData(df, input_size, output_size)
    if is_training:
        df = dataflow.PrefetchDataZMQ(df, nr_proc=8)
    df = dataflow.BatchData(df, batch_size, use_list=True)
    df = dataflow.RepeatedData(df, -1)
    if test_speed:
        dataflow.TestDataSpeed(df, size=1000).start()
    df.reset_state()
    return df, size

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

#/shared/kgco-research/mil/harish/pcn_data/pcd_datav2/voxels/sofa/sofa_vxl_7.npz'
#training_list ---> contains the list of all training samples
def load_training_data(indices, training_list, pt_dir, stage=1, npz=False):
    #inputs, gt, coarse, middle
    in_lst = []
    vx_lst = []
    gt_lst = []
    coarse_lst = []
    middle_lst = []
    # pdb.set_trace()
    for indx in indices:
        nme = training_list[indx]
        cat = nme.split('_')[0]
        indc = nme.split('_')[1].split('.')[0]
        if npz:
            #add path to training_list[indx]
            #nme = training_list[indx]
            #dt_path = pt_dir + training_list[indx]
            sampled_path = os.path.join(pt_dir, 'sampled_data', training_list[indx])
            npz = np.load(sampled_path)
            inpt = npz['inputs']
            gt = npz['gt']
            coarse = npz['coarse']
            middle = npz['middle']
            #ex = 'airplane_1.npz'
            vxl_fl = os.path.join(pt_dir, 'voxels', cat)
            vxl = np.load(vxl_fl+'/'+cat+'_vxl_'+str(indc)+'.npz')['voxels']
            if stage==1:
                in_lst.append(inpt)
                gt_lst.append(coarse)
            elif stage==2:
                in_lst.append(coarse)
                gt_lst.append(middle)
            else:
                in_lst.append(inpt)
                gt_lst.append(gt)
                coarse_lst.append(coarse)
                middle_lst.append(middle)
            vx_lst.append(vxl)
        else:
            sample = nme.replace('npz','pcd')
            gt_smple = cat+'_'+indc+'_gt'+'.pcd'
            vxl = cat+'_vxl_'+indc+'.npz'
            sample_pth = os.path.join(pt_dir,cat,sample)
            gt_pth = os.path.join(pt_dir,cat,gt_smple)
            vxl_pth = os.path.join(pt_dir,'voxels',cat,vxl)
            in_arr = read_pcd(sample_pth)
            gt_arr = read_pcd(gt_pth)
            vxl_arr = np.load(vxl_pth)['voxels']
            in_lst.append(in_arr)
            gt_lst.append(gt_arr)
            vx_lst.append(vxl_arr)
    # pdb.set_trace()
    inputs = np.squeeze(np.array(in_lst))
    gt = np.squeeze(np.array(gt_lst))
    coarse = np.squeeze(np.array(coarse_lst))
    middle = np.squeeze(np.array(middle_lst))
    voxels = np.squeeze(np.array(vx_lst))
    training_ids = [training_list[indx].replace('.npz','') for indx in indices]
    # pdb.set_trace()
    # print('voxels shape is {}'.format(voxels.shape))
    return inputs, gt, coarse, middle,  voxels, training_ids

def load_training_data_stage2(indices, training_list, pt_dir):
    #inputs, gt, coarse, middle
    in_lst = []
    feat_lst = []
    gt_lst = []
    # pdb.set_trace()
    for indx in indices:
        nme = training_list[indx]
        cat = nme.split('_')[0]
        indc = nme.split('_')[1].split('.')[0]
        #add path to training_list[indx]
        #nme = training_list[indx]
        #dt_path = pt_dir + training_list[indx]
        sampled_path = os.path.join(pt_dir, 'sampled_data', training_list[indx])
        coarsein_path = os.path.join(pt_dir, 'stage_1', training_list[indx])
        feat_pth = os.path.join(pt_dir, 'stage_1', cat+'_'+str(indc)+'_global.npz')
        npz = np.load(sampled_path)
        in_npz = np.load(coarsein_path)
        feat_npz = np.load(feat_pth)
        inpt = in_npz['input']
        feat = feat_npz['input']
        gt = npz['middle']
        in_lst.append(inpt)
        feat_lst.append(feat)
        gt_lst.append(gt)

    # pdb.set_trace()
    inputs = np.squeeze(np.array(in_lst))
    gt = np.squeeze(np.array(gt_lst))
    features = np.squeeze(np.array(feat_lst))
    training_ids = [training_list[indx].replace('.npz','') for indx in indices]
    # pdb.set_trace()
    # print('voxels shape is {}'.format(voxels.shape))
    return inputs, gt, features, training_ids


def load_training_data_stage3(indices, training_list, pt_dir):
    #inputs, gt, coarse, middle
    in_lst = []
    feat_lst = []
    coarse_lst = []
    gt_lst = []
    # pdb.set_trace()
    for indx in indices:
        nme = training_list[indx]
        cat = nme.split('_')[0]
        indc = nme.split('_')[1].split('.')[0]
        #add path to training_list[indx]
        #nme = training_list[indx]
        #dt_path = pt_dir + training_list[indx]
        sampled_path = os.path.join(pt_dir, 'sampled_data', training_list[indx])
        coarsein_path = os.path.join(pt_dir, 'stage_1', training_list[indx])
        feat_pth = os.path.join(pt_dir, 'stage_1', cat+'_'+str(indc)+'_global.npz')
        middle_path = os.path.join(pt_dir, 'stage_2', training_list[indx])
        npz = np.load(sampled_path)
        in_npz = np.load(middle_path)
        feat_npz = np.load(feat_pth)
        coarse_npz = np.load(coarsein_path)
        inpt = in_npz['input']
        feat = feat_npz['input']
        coarse = coarse_npz['input']
        gt = npz['gt']
        in_lst.append(inpt)
        coarse_lst.append(coarse)
        feat_lst.append(feat)
        gt_lst.append(gt)

    # pdb.set_trace()
    inputs = np.squeeze(np.array(in_lst))
    gt = np.squeeze(np.array(gt_lst))
    features = np.squeeze(np.array(feat_lst))
    coarse_in = np.squeeze(np.array(coarse_lst))
    training_ids = [training_list[indx].replace('.npz','') for indx in indices]
    # pdb.set_trace()
    # print('voxels shape is {}'.format(voxels.shape))
    return inputs, gt, features, coarse_in, training_ids


def plot_loss(losses):
    plt.figure(figsize=(16,9))
    plt.plot(losses)
    plt.title('Training losses')


def loadDataFile(filename):
    return load_h5(filename)


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)


def save_pcd(filename, points):
    pcd = PointCloud()
    pcd.points = Vector3dVector(points)
    write_point_cloud(filename, pcd)


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data
    

if __name__=='__main__':
    trn_fl = open('train_lst.txt')
    train_list = trn_fl.readlines()
    train_list = [smpl[:-1] for smpl in train_list]
    training_indx = np.arange(0,len(train_list))
    nm_btchs = int(len(training_indx)/32)
    for btch in range(nm_btchs):
        str_indx = btch*32
        end_indx = (btch+1)*32
        if end_indx>len(training_indx):
            # btch_indx = training_indx[str_indx:]
            continue
        else:
            btch_indx = training_indx[str_indx:end_indx]
        inputs, gt, features, ids = load_training_data_stage2(btch_indx, train_list, '/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/')
        # pdb.set_trace()
        if not len(features.shape) is 2:
            pdb.set_trace()
        else:
            continue