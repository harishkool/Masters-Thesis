''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import glob
import pdb
import numpy as np
from open3d import *


def read_pcd(filename):
    pcd = read_point_cloud(filename)
    return np.array(pcd.points)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library('tf_sampling.so')
def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]
    
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')
    



def generate_samples(pcd_dir, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    cat = os.listdir(pcd_dir)
    # pdb.set_trace()
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    cat = ["table"]
    for dir in cat: 
        cat_dir = os.path.join(pcd_dir,dir)
        all_pcd = glob.glob(cat_dir+'/*.pcd')
        print(all_pcd[-1])
        tot_cnt = int(len(all_pcd)/2)
        for cnt in range(1,tot_cnt+1):
            in_pcd = dir+'_'+str(cnt)+'.pcd'
            gt_pcd = dir+'_'+str(cnt)+'_gt.pcd'
            in_arr = read_pcd(os.path.join(cat_dir,in_pcd))
            gt_arr = read_pcd(os.path.join(cat_dir,gt_pcd))
            gt_arr = gt_arr.astype(np.float32)
            in_arr = in_arr.astype(np.float32)
            # pdb.set_trace()
            in_arr = np.expand_dims(in_arr,0)
            gt_arr = np.expand_dims(gt_arr,0)
            # gt_tensr = tf.constant(gt_arr, dtype = tf.float32)
            feed_dict = {inputs_pl: gt_arr}
            coarse_pnts = farthest_point_sample(1024,inputs_pl)
            coarse_pcd = gather_point(inputs_pl,coarse_pnts)
            middle_pnts = farthest_point_sample(4096, inputs_pl)
            middle_pcd = gather_point(inputs_pl,middle_pnts)
            coarse, middle = sess.run([coarse_pcd, middle_pcd], feed_dict=feed_dict)
            new_pcd = os.path.join(save_dir, in_pcd.split('.')[0]+'.npz')
            print(cnt)
            if not os.path.isfile(new_pcd):
                np.savez(new_pcd,inputs=in_arr,gt=gt_arr,coarse=coarse,middle=middle)


if __name__=='__main__':
    generate_samples('/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/','/shared/kgcoe-research/mil/harish/pcn_data/sampled_data/')



# if __name__=='__main__':
#     import numpy as np
#     np.random.seed(100)
#     triangles=np.random.rand(1,5,3,3).astype('float32')
#     with tf.device('/gpu:1'):
#         inp=tf.constant(triangles)
#         tria=inp[:,:,0,:]
#         trib=inp[:,:,1,:]
#         tric=inp[:,:,2,:]
#         areas=tf.sqrt(tf.reduce_sum(tf.cross(trib-tria,tric-tria)**2,2)+1e-9)
#         randomnumbers=tf.random_uniform((1,8192))
#         triids=prob_sample(areas,randomnumbers)
#         tria_sample=gather_point(tria,triids)
#         # trib_sample=gather_point(trib,triids)
#         # tric_sample=gather_point(tric,triids)
#         # us=tf.random_uniform((1,8192))
#         # vs=tf.random_uniform((1,8192))
#         # uplusv=1-tf.abs(us+vs-1)
#         # uminusv=us-vs
#         # us=(uplusv+uminusv)*0.5
#         # vs=(uplusv-uminusv)*0.5
#         # pt_sample=tria_sample+(trib_sample-tria_sample)*tf.expand_dims(us,-1)+(tric_sample-tria_sample)*tf.expand_dims(vs,-1)
#         # print 'pt_sample: ', pt_sample
#         # reduced_sample=gather_point(pt_sample,farthest_point_sample(1024,pt_sample))
#         # print reduced_sample
#     with tf.Session('') as sess:
#         ret=sess.run(tria_sample)
#         print(ret.shape)
    # print ret.shape,ret.dtype
    # import cPickle as pickle
    # pickle.dump(ret,open('1.pkl','wb'),-1)
