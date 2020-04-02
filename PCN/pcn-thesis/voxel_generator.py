import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append()
# sys.path.append(os.path.join(BASE_DIR, 'ops_utils'))
sys.path.append('../')
import numpy as np
from point_cloud_ops import points_to_voxel
import pandas as pd
import pdb
from util import read_pcd, save_pcd, lmdb_dataflow
# from data_util import lmdb_dataflow, get_queued_data

class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        # pdb.set_trace()
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, inputs, max_voxels):
        voxel_list = []
        for i in range(len(inputs)):
            voxels, coors, num_points_per_voxel = points_to_voxel(
                    inputs[i], self._voxel_size, self._point_cloud_range,
                    self._max_num_points, True, max_voxels)
            voxel_list.append(voxels)
        max_shp = max([x.shape[0] for x in voxel_list])
        mdf_voxellist = []
        # pdb.set_trace()
        for voxels in voxel_list:
            diff = max_shp - voxels.shape[0]
            zeros = np.zeros((diff, self._max_num_points, 3))
            voxels_mdfd = np.concatenate([voxels, zeros])
            mdf_voxellist.append(voxels_mdfd)
        voxel_stack = np.stack(mdf_voxellist)
        return voxel_stack

    def preprocess(self, lidar):

        # shuffling the points
        np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array([self._point_cloud_range[0], self._point_cloud_range[1], self._point_cloud_range[2]])) / (
                        self._voxel_size[0], self._voxel_size[1], self._voxel_size[2])).astype(np.int32)

        # convert to  (D, H, W)
        voxel_coords = voxel_coords[:,[2,1,0]]
        voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, \
                                                  return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros((self._max_num_points, 7), dtype=np.float32)
            pts = lidar[inv_ind == i]
            if voxel_counts[i] > self._max_num_points:
                pts = pts[:self._max_num_points, :]
                voxel_counts[i] = self._max_num_points
            # augment the points
            voxel[:pts.shape[0], :] = np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
            voxel_features.append(voxel)
        return np.array(voxel_features), voxel_coords


    def get_filtered_lidar(self, lidar):

        pxs = lidar[:, 0]
        pys = lidar[:, 1]
        pzs = lidar[:, 2]

        filter_x = np.where((pxs >= self._point_cloud_range[0]) & (pxs < self._point_cloud_range[3]))[0]
        filter_y = np.where((pys >= self._point_cloud_range[1]) & (pys < self._point_cloud_range[4]))[0]
        filter_z = np.where((pzs >= self._point_cloud_range[2]) & (pzs < self._point_cloud_range[5]))[0]
        filter_xy = np.intersect1d(filter_x, filter_y)
        filter_xyz = np.intersect1d(filter_xy, filter_z)

        return lidar[filter_xyz]

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


#  voxel_generator {
#       point_cloud_range : [0, -40, -3, 70.4, 40, 1]
#       # point_cloud_range : [0, -32.0, -3, 52.8, 32.0, 1]
#       voxel_size : [0.05, 0.05, 0.1]
#       max_number_of_points_per_voxel : 5
#     }

if __name__ == "__main__":
    voxgen = VoxelGenerator([0.05, 0.05, 0.05],[-0.2, -0.15, -0.25, 0.2, 0.2, 0.2], 30)
    df_train, num_train = lmdb_dataflow(
        '/shared/kgcoe-research/mil/harish/pcn_data/data/shapenet/train.lmdb', 4, 2048, 16384, is_training=True)
    # pdb.set_trace()
    train_gen = df_train.get_data()
    ids, inputs, gt = next(train_gen)
    # pc = inputs[0]
    # pcd = np.copy(pc)
    # save_pcd('pcn_train1.pcd', pcd)
    voxel_stack  = voxgen.generate(inputs, max_voxels=2000)
    pdb.set_trace()
    print(voxel_stack.shape)

