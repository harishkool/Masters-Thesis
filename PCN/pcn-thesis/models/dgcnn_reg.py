import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from tf_util import add_train_summary, add_valid_summary
from transform_nets import input_transform_net
import pdb

def get_model(point_cloud, is_training, indx, bn_decay=None):
  """ Classification PointNet, input is BxNx3"""
  # pdb.set_trace()
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, indx, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, transform)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d_reg(edge_feature, 64, kernel_size=[1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1'+str(indx), bn_decay=bn_decay)
  # print('Net shape is: {}'.format(net.get_shape()))
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  # print('Net shape after max is: {}'.format(net.get_shape()))
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d_reg(edge_feature, 64, kernel_size=[1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2'+str(indx), bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net
 
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

  net = tf_util.conv2d_reg(edge_feature, 64, kernel_size=[1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3'+str(indx), bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  net = tf_util.conv2d_reg(edge_feature, 128, kernel_size=[1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4'+str(indx), bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 320, 1024, kernel_size=[1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg'+str(indx), bn_decay=bn_decay)
 
  net = tf.reduce_max(net, axis=1, keep_dims=True) 

  return net 












