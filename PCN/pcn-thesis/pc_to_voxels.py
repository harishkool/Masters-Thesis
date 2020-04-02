import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import binvox_rw
from util import save_pcd, lmdb_dataflow

df_train, num_train = lmdb_dataflow(
    '/shared/kgcoe-research/mil/harish/pcn_data/data/shapenet/train.lmdb', 1, 2048, 16384, is_training=True)
train_gen = df_train.get_data()
for i in range(500):
    ids, inputs, gt = next(train_gen)
    pdd = pd.DataFrame({"x":inputs[0][:,0],"y":inputs[0][:,1],"z":inputs[0][:,2]})
    cloud = PyntCloud(pdd)
    # cloud = PyntCloud.from_file("test/00000.txt",
    #                             sep=" ",
    #                             header=0,
    #                             names=["x","y","z"])

    # cloud.plot(mesh=True, backend="threejs")

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=32, n_y=32, n_z=32)
    voxelgrid = cloud.structures[voxelgrid_id]
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((32, 32, 32)).astype(np.bool)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True
    fl_name = 'new_data/test'+str(i+1)
    pc = inputs[0]
    pcd = np.copy(pc)
    save_pcd(fl_name+'.pcd', pcd)
    with open(fl_name+".binvox", 'wb') as f:
        v = binvox_rw.Voxels(voxel, (32, 32, 32), (0, 0, 0), 1, 'xyz')
        v.write(f)