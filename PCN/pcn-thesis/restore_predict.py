# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

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
from voxel_generator import *
from util import load_training_data, save_pcd
import pdb
import shutil


def restore(args):
    # pdb.set_trace()
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    inputs_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_input_points, 3), 'inputs')
    voxels_pl = tf.placeholder(tf.float32, (args.batch_size, 70, 40, 3))
    npts_pl = tf.placeholder(tf.int32, (None), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    # model = model_module.Model(inputs_pl, gt_pl, alpha, beta, args)
    model = model_module.Model(inputs_pl, gt_pl, voxels_pl)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')
    # trainer = tf.train.RMSPropOptimizer(learning_rate)
    trainer = tf.train.AdamOptimizer(learning_rate)

    # df_train, num_train = lmdb_dataflow(
    #     args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    # train_gen = df_train.get_data()
    # df_valid, num_valid = lmdb_dataflow(
    #     args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    # valid_gen = df_valid.get_data()
    # pdb.set_trace()
    # pdb.set_trace()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
    training_fl = open('train_lst.txt')
    training_list = training_fl.readlines()
    training_list = [smp[:-1] for smp in training_list]
    valid_fl = open('valid_lst.txt')
    valid_list = valid_fl.readlines()
    valid_list = [smp[:-1] for smp in valid_list]
    training_indx = np.arange(0,len(training_list))
    np.random.shuffle(training_indx)
    nm_btchs = int(len(training_indx)/args.batch_size)
    valid_indx = np.arange(0,len(valid_list))
    num_valid = len(valid_list)

    save_dir = os.path.join(args.lmdb_train,'stage_'+str(args.stage))
    # # if not  os.path.isdir(save_dir):
    # #     os.mkdir(save_dir)
    # # else:
    # #     shutil.rmtree(save_dir)
    # #     os.mkdir(save_dir)
    # pdb.set_trace()
    for indx in training_indx:
        btch_indx = [indx]
        nme = training_list[indx]
        pre_nme = nme.split('.')[0]
        fin_nme = pre_nme+'_global'+'.npz'
        pcd_pth = os.path.join(save_dir, fin_nme)
        coarse_pth = os.path.join(save_dir, nme)
        inputs, gt, voxels, ids = load_training_data(btch_indx, training_list, args.lmdb_train, args.stage, True) 
        inputs = np.expand_dims(inputs,0) 
        gt = np.expand_dims(gt,0) 
        voxels = np.expand_dims(voxels,0) 
        feed_dict = {inputs_pl: inputs, voxels_pl:voxels, gt_pl: gt, is_training_pl: False} #add voxels also here
        coarse_out, pcd_out = sess.run([model.coarse, model.features], feed_dict=feed_dict)
        coarse_out = coarse_out[0]
        pcd_out = pcd_out[0]
        #car_66.npz
        np.savez(pcd_pth, input=pcd_out)
        np.savez(coarse_pth, input=coarse_out)


    # pdb.set_trace()
    # save_dir = os.path.join(args.lmdb_valid, 'stage_'+str(args.stage))
    # if not  os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # else:
    #     shutil.rmtree(save_dir)
    #     os.mkdir(save_dir)

    # for indx in valid_indx:
    #     btch_indx = [indx]
    #     inputs, gt, voxels, ids = load_training_data(btch_indx, valid_list, args.lmdb_valid, args.stage, True) 
    #     inputs = np.expand_dims(inputs,0) 
    #     gt = np.expand_dims(gt,0) 
    #     voxels = np.expand_dims(voxels,0) 
    #     feed_dict = {inputs_pl: inputs, voxels_pl:voxels, gt_pl: gt, is_training_pl: False} #add voxels also here
    #     coarse_out, pcd_out = sess.run([model.coarse, model.features], feed_dict=feed_dict)
    #     pcd_out = pcd_out[0]
    #     nme = valid_list[indx]
    #     pre_nme = nme.split('.')[0]
    #     fin_nme = pre_nme+'_global'+'.npz'
    #     pcd_pth = os.path.join(save_dir, fin_nme)
    #     coarse_pth = os.path.join(save_dir, nme)
    #     np.savez(pcd_pth, input=pcd_out)
    #     np.savez(coarse_pth, input=coarse_out)

    # for btch in range(num_eval_steps):
    #     # epoch = step * args.batch_size // num_train + 1
    #     str_indx = btch*args.batch_size
    #     end_indx = (btch+1)*args.batch_size
    #     if end_indx>len(valid_indx):
    #         btch_indx = valid_indx[str_indx:len(valid_indx)]
    #         add = end_indx - len(valid_indx)
    #         lst = [i for i in range(add)]
    #         btch_indx = btch_indx + lst
    #         # continue
    #     else:
    #         btch_indx = valid_indx[str_indx:end_indx]
    #     inputs, gt, voxels, ids = load_training_data(btch_indx, valid_list, args.lmdb_valid, args.stage, True) 
    #     feed_dict = {inputs_pl: inputs, voxels_pl:voxels, gt_pl: gt, is_training_pl: False} #add voxels also here
    #     pcd_out = sess.run([model.coarse], feed_dict=feed_dict)
    #     for i,indx in enumerate(btch_indx):
    #         nme = valid_list[indx]
    #         pcd_pth = os.path.join(save_dir,nme)
    #         np.savez(pcd_pth, input=pcd_out[i,:,:])

    sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/')
    parser.add_argument('--lmdb_valid', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid/')
    parser.add_argument('--log_dir', default='log/pcn_emd_stage1')
    parser.add_argument('--model_type', default='pcn_emd_stage1')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--num_gt_points', type=int, default=1024)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    args = parser.parse_args()
    restore(args)


# GT shape is (32, 16384, 3)
# Features shape is (32, 2048, 256)
# Input shape is (32, 2048, 3)
# Features global shape is (32, 1, 256)
# Features after concat shape is (32, 2048, 512)
# Final features shape is (32, 1024)
# Grid feat shape is (32, 16384, 2)
# Point feat shape is (32, 16384, 3)
# Global  feat shape is (32, 16384, 1024)
# feat shape is (32, 16384, 1029)