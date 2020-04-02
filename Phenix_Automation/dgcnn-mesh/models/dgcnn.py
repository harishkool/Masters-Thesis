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
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
  # pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  # labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, None, 3))

  labels_pl = tf.placeholder(tf.int32, shape=(None))
  
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, var, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  # adj_matrix = tf_util.pairwise_distance(point_cloud)
  # nn_idx = tf_util.knn(adj_matrix, k=k)
  # edge_feature = tf_util.get_edge_feature(point_cloud, 3, var, nn_idx=nn_idx, k=k)

  # with tf.variable_scope('transform_net1') as sc:
    # transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  # point_cloud_transformed = tf.matmul(point_cloud, transform)
  # point_cloud = tf.expand_dims(point_cloud, 0)
  point_cloud = tf.reshape(point_cloud,[1,-1,3])
  point_cloud_transformed = point_cloud
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  # print('pc transformed shape is {}'.format(point_cloud_transformed.get_shape()))
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, 3, var, nn_idx=nn_idx, k=k)
  # print('edge shape 1 is {}'.format(edge_feature.get_shape()))
  net = tf_util.conv2d(edge_feature, 6, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  # print('net 1 shape is {}'.format(net.get_shape()))
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net
  net = tf.reshape(net,[1,-1,64])
  # print('net 1 shape aftr maxpool is {}'.format(net.get_shape()))
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, 64, var, nn_idx=nn_idx, k=k)
  
  net = tf_util.conv2d(edge_feature, 128, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net
  net = tf.reshape(net,[1,-1,64])

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, 64, var, nn_idx=nn_idx, k=k)  

  net = tf_util.conv2d(edge_feature, 128, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net
  net = tf.reshape(net,[1,-1,64])

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, 64, var, nn_idx=nn_idx, k=k)  
  
  net = tf_util.conv2d(edge_feature, 128, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net
  net = tf.reshape(net,[1,-1,128])

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 320, 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)
 
  # net = tf.reduce_max(net, axis=1, keep_dims=True) 
  # net = tf.squeeze(net)
  print('final net shape is {}'.format(net.get_shape()))
  # MLP on global point cloud vector
  # net = tf.reshape(net, [batch_size, -1]) 
  # net = tf_util.fully_connected(net, 1024, 512, bn=True, is_training=is_training,
  #                               scope='fc1', bn_decay=bn_decay)
  # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
  #                        scope='dp1')
  # net = tf_util.fully_connected(net, 512, 256, bn=True, is_training=is_training,
  #                               scope='fc2', bn_decay=bn_decay)
  # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
  #                       scope='dp2')
  # net = tf_util.fully_connected(net, 256, 40, activation_fn=None, scope='fc3')

  net = tf_util.conv2d(net, 1024, 256, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn5', bn_decay=bn_decay)


  net = tf_util.conv2d(net, 256, 256, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn6', bn_decay=bn_decay)

  net = tf_util.conv2d(net, 256, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn7', bn_decay=bn_decay)


  net = tf_util.conv2d(net, 128, 2, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='seg', bn_decay=bn_decay)

  net = tf.squeeze(net)

  return net, end_points


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  # labels = tf.one_hot(indices=label, depth=40)
  labels = label
  # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      # print res1.shape
      # print res

      # print res2.shape
      # print res2











