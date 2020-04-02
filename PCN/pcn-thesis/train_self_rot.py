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
from util import load_training_data, rotate_point_cloud_by_angle
import pdb
from models import pcn_dgcnn_self
import clr


BASE_LEARNING_RATE = 0.01
DECAY_STEP = 100000
DECAY_RATE = 0.7

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_learning_rate(batch, args):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * args.batch_size,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch, args):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*args.batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


#total train : 206370
#batch size :32
#total steps : (206370/32)*50 = 322453
def train(args):
    # pdb.set_trace()
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    
    inputs_pl = tf.placeholder(tf.float32, ((args.batch_size)*4, 2048, 3), 'inputs')
    # voxels_pl = tf.placeholder(tf.float32, (args.batch_size, 70, 40, 3))
    gt_pl = tf.placeholder(tf.int32, ((args.batch_size)*4), 'ground_truths')
    batch = tf.Variable(0,trainable=False)
    bn_decay = get_bn_decay(batch, args)
    # model = model_module.Model(inputs_pl, gt_pl, alpha, beta, args)
    pred, end_points = pcn_dgcnn_self.get_model(inputs_pl, is_training_pl, bn_decay)
    # pdb.set_trace()
    loss = pcn_dgcnn_self.get_loss(pred, gt_pl)
    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')
    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')
    # trainer = tf.train.RMSPropOptimizer(learning_rate)
    # trainer = tf.train.AdamOptimizer(learning_rate=clr.cyclic_learning_rate(global_step=global_step, mode='triangular2'))
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(loss, global_step)

    # df_train, num_train = lmdb_dataflow(
    #     args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    # train_gen = df_train.get_data()
    # df_valid, num_valid = lmdb_dataflow(
    #     args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    # valid_gen = df_valid.get_data()
    pdb.set_trace()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
    else:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.log_dir):
            # delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
            #                            % args.log_dir, 'white', 'on_red'))
            # if delete_key == 'y' or delete_key == "":
            os.system('rm -rf %s/*' % args.log_dir)
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
        # os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))  # bkp of model def
        # os.system('cp train.py %s' % args.log_dir)                         # bkp of train procedure
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    # pdb.set_trace()
    training_fl = open('train_less_lst.txt')
    training_list = training_fl.readlines()
    training_list = [smp[:-1] for smp in training_list]
    valid_fl = open('valid_lst.txt')
    valid_list = valid_fl.readlines()
    valid_list = [smp[:-1] for smp in valid_list]
    total_time = 0
    train_start = time.time()
    training_indx = np.arange(0,len(training_list))
    np.random.shuffle(training_indx)
    nm_btchs = int(len(training_indx)/args.batch_size)
    valid_indx = np.arange(0,len(valid_list))
    num_valid = len(valid_list)
    # for step in range(init_step+1, args.max_step+1):
    init_step = sess.run(global_step)
    # step = 0
    # pdb.set_trace()
    for epoch in range(args.num_epochs):
        total_loss = 0
        for btch in range(nm_btchs):
            # step += 1
            # epoch = step * args.batch_size // num_train + 1
            str_indx = btch*args.batch_size
            end_indx = (btch+1)*args.batch_size
            if end_indx>len(training_indx):
                # btch_indx = training_indx[str_indx:]
                continue
            else:
                btch_indx = training_indx[str_indx:end_indx]
            # ids, inputs, gt = next(train_gen)
            #inpts shape is bn x num_pts x 3
            #voxels shape [bn=1, 50, 50, 3] for example
            #vfe should return 1, 50, features 
            #voxels_pl: 1, 50, 50, 3
            #genrlize: bn, num_voxels, max_num_points, 3
            # voxgen = VoxelGenerator([0.05, 0.05, 0.05],[-0.2, -0.15, -0.25, 0.2, 0.2, 0.2], 50)
            # voxels  = voxgen.generate(inputs, max_voxels=2000)
            # print('voxels shape is {}'.format(voxels.shape))
            # pdb.set_trace()
            inputs, gt, coarse, middle, voxels, ids = load_training_data(btch_indx, training_list, args.lmdb_train)
            inputs_1 = rotate_point_cloud_by_angle(inputs, 0)
            labels_1 = np.zeros((args.batch_size)) 
            # inputs_2 = rotate_point_cloud_by_angle(inputs, np.pi/2)
            inputs_2 = rotate_point_cloud_by_angle(inputs, 1*(np.pi/180))
            labels_2 = np.ones((args.batch_size))  
            inputs_3 = rotate_point_cloud_by_angle(inputs, 2*(np.pi/180)) 
            labels_3 = np.ones((args.batch_size)) * 2 
            inputs_4 = rotate_point_cloud_by_angle(inputs, 3*(np.pi/180)) 
            labels_4 = np.ones((args.batch_size)) * 3 
            start = time.time()
            inputs_fn = np.concatenate([inputs_1, inputs_2, inputs_3, inputs_4])
            labels_fn = np.concatenate([labels_1, labels_2, labels_3, labels_4])
            # pdb.set_trace()
            train_indx = np.arange(0,len(inputs_fn))
            np.random.shuffle(train_indx)
            inputs_fn = inputs_fn[train_indx]
            labels_fn = labels_fn[train_indx]
            nm_points = inputs_fn.shape[1]
            # pdb.set_trace()
            feed_dict = {inputs_pl: inputs_fn, gt_pl: labels_fn, is_training_pl: True} 
            # feed_dict = {inputs_pl: inputs, gt_pl: gt, is_training_pl: True}
            _, loss_val, step, _, summary, pred_val = sess.run([train_op, loss, global_step, batch, train_summary, pred], feed_dict=feed_dict)
            
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val==labels_fn)
            accuracy = correct/ float((args.batch_size)*4)
            tf.summary.scalar('accuracy', accuracy)

            total_loss  += loss_val
            total_time += time.time() - start
            writer.add_summary(summary, step)
            # pdb.set_trace()
            if step % args.steps_per_print == 0:
                print('epoch %d  step %d  loss %.8f  accuracy %.8f - time per batch %.4f' %
                    (epoch, step, total_loss/args.steps_per_print, accuracy, total_time / args.steps_per_print))
                total_loss = 0
                total_time = 0
        # pdb.set_trace()               
        if epoch % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            total_eval_loss = 0
            total_eval_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
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
                inputs, gt, coarse, middle, voxels, ids = load_training_data(btch_indx, valid_list, args.lmdb_valid)
                inputs_1 = rotate_point_cloud_by_angle(inputs, 0)
                labels_1 = np.zeros((args.batch_size)) 
                inputs_2 = rotate_point_cloud_by_angle(inputs, np.pi/2)
                labels_2 = np.ones((args.batch_size))  
                inputs_3 = rotate_point_cloud_by_angle(inputs, np.pi) 
                labels_3 = np.ones((args.batch_size)) * 2 
                inputs_4 = rotate_point_cloud_by_angle(inputs, np.pi*(3/2)) 
                labels_4 = np.ones((args.batch_size)) * 3 
                inputs_fn = np.concatenate([inputs_1, inputs_2, inputs_3, inputs_4])
                labels_fn = np.concatenate([labels_1, labels_2, labels_3, labels_4])
                train_indx = np.arange(0,len(inputs_fn))
                np.random.shuffle(train_indx)
                inputs_fn = inputs_fn[train_indx]
                labels_fn = labels_fn[train_indx]
                nm_points = inputs_fn.shape[1]
                feed_dict = {inputs_pl: inputs_fn, gt_pl: labels_fn, is_training_pl: False} 
                loss_val, pred_val = sess.run([loss, pred], feed_dict=feed_dict)
                pred_val = np.argmax(pred_val, 1)
                correct = np.sum(pred_val == labels_fn)
                accuracy = correct/ float(args.batch_size)

                total_eval_loss += loss_val
                total_eval_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, epoch)
            print(colored('epoch %d  step %d  loss %.8f accuracy %.8f- time per batch %.4f' %
                        (epoch, step, total_loss / num_eval_steps, accuracy, total_time / num_eval_steps),
                        'grey', 'on_green'))

        if epoch % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/')
    parser.add_argument('--lmdb_valid', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid/')
    parser.add_argument('--log_dir', default='log/pcn_capsule')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=2048)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    # parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=10)
    parser.add_argument('--steps_per_save', type=int, default=25)
    args = parser.parse_args()
    train(args)


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