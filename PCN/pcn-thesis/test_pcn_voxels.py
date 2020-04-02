import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views
# from voxel_generator import *
from util import load_training_data, rotate_point_cloud_by_angle
import pdb
import numpy as np

# Total loss 52.94661596 loss 0.07607272 -- pcn_emd_voxels_new
# Total loss 52.71563032 loss 0.07574085 -- pcn_emd_voxels
# Total loss 9.06075988 loss 0.01301833 -- only complete point clouds -- pcn_emd_voxels
# Total loss 43.47772792 loss 0.06211104 -- EMD distance -- only complete 
# Total loss 43.83404756 loss 0.06262007 -- EMD --coarse

def test(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, 2048, 3), 'inputs')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')
    coarse_pl = tf.placeholder(tf.float32, (args.batch_size, 1024, 3), 'ground_truths')
    beta = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    voxels_pl = tf.placeholder(tf.float32, (args.batch_size, 70, 40, 3))
    middle_pl = tf.placeholder(tf.float32, (args.batch_size, 4096, 3), 'ground_truths')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl,  gt_pl, voxels_pl, coarse_pl, middle_pl, alpha, beta)
    # pdb.set_trace()
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
    valid_fl = open('valid_lst.txt')
    valid_list = valid_fl.readlines()
    valid_list = [smp[:-1] for smp in valid_list]
    valid_indx = np.arange(0,len(valid_list))
    num_valid = len(valid_list)
    num_eval_steps = num_valid // args.batch_size
    total_eval_loss = 0
    sess.run(tf.local_variables_initializer())
    total_seen = 0
    for i in range(num_eval_steps):
        # pdb.set_trace()
        # ids, inputs, gt = next(valid_gen)
        # feed_dict = {inputs_pl: inputs, voxels_pl:voxels, gt_pl: gt, is_training_pl: False}
        # feed_dict = {inputs_pl: inputs, gt_pl: gt, is_training_pl: False}
        str_indx = i*args.batch_size
        end_indx = (i+1)*args.batch_size
        if end_indx>len(valid_indx):
            btch_indx = valid_indx[str_indx:]
            # continue
        else:
            btch_indx = valid_indx[str_indx:end_indx]
        inputs, gt, coarse, middle, voxels, ids = load_training_data(btch_indx, valid_list, args.lmdb_valid, 0, True)
        feed_dict = {inputs_pl: inputs, gt_pl: gt, coarse_pl: coarse, middle_pl:middle, voxels_pl: voxels, is_training_pl: False} 
        loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)
        total_eval_loss += loss*args.batch_size
        total_seen += args.batch_size
    print('Total loss %.8f loss %.8f'% (total_eval_loss, total_eval_loss / total_seen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_valid', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid/')
    parser.add_argument('--log_dir', default='log/trained_models/pcn_cd')
    parser.add_argument('--model_type', default='pcn_original')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    args = parser.parse_args()
    test(args)

