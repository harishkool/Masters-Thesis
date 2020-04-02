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
from util import load_training_data, save_pcd, load_training_data_stage3
import pdb
import shutil

#total train : 206370
#batch size :32
#total steps : (206370/32)*50 = 322453
def train(args):
    # pdb.set_trace()
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    coarse_pl = tf.placeholder(tf.float32, (args.batch_size, 1024, 3), 'coarse')
    middle_pl = tf.placeholder(tf.float32, (args.batch_size, 4096, 3), 'middle')
    features_pl = tf.placeholder(tf.float32, (args.batch_size, 1024))
    npts_pl = tf.placeholder(tf.int32, (None), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, 16384, 3), 'ground_truths')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    # model = model_module.Model(inputs_pl, gt_pl, alpha, beta, args)
    model = model_module.Model(features_pl, coarse_pl, middle_pl, gt_pl)

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
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    # df_train, num_train = lmdb_dataflow(
    #     args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    # train_gen = df_train.get_data()
    # df_valid, num_valid = lmdb_dataflow(
    #     args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    # valid_gen = df_valid.get_data()
    # pdb.set_trace()
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
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))  # bkp of model def
        os.system('cp train.py %s' % args.log_dir)                         # bkp of train procedure
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    # pdb.set_trace()
    training_fl = open('train_lst.txt')
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
    step = 0
    ids_g = None
    for epoch in range(args.num_epochs):
        total_loss = 0
        for btch in range(nm_btchs):
            step += 1
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
            inputs, gt, features, coarse_in, ids = load_training_data_stage3(btch_indx, training_list, args.lmdb_train)
            ids_g = ids
            # pdb.set_trace()
            start = time.time()
            if(not (len(features.shape)==2) or (not(len(inputs.shape)==3)) or (not(len(gt.shape)==3))):
                pdb.set_trace()
            # print('Inputs: '+str(inputs.shape)+', gt: '+str(gt.shape)+', features: '+str(features.shape))
            feed_dict = {middle_pl: inputs, coarse_pl:coarse_in, 
                features_pl:features, gt_pl: gt, is_training_pl: True} #add voxels also here
            # feed_dict = {inputs_pl: inputs, gt_pl: gt, is_training_pl: True}
            _, loss, summary = sess.run([train_op, model.loss, train_summary], feed_dict=feed_dict)
            total_loss  += loss
            total_time += time.time() - start
            writer.add_summary(summary, step)
            
            if step % args.steps_per_print == 0:
                print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                    (epoch, step, total_loss/args.steps_per_print, total_time / args.steps_per_print))
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
                inputs, gt, features, coarse_in, ids = load_training_data_stage3(btch_indx, valid_list, args.lmdb_valid)
                feed_dict = {middle_pl: inputs, features_pl:features, coarse_pl:coarse_in, gt_pl: gt, is_training_pl: False} #add voxels also here
                loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)
                total_eval_loss += loss
                total_eval_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, epoch)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                        (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                        'grey', 'on_green'))

        if epoch % args.steps_per_visu == 0:
            all_pcds = sess.run(model.visualize_ops, feed_dict=feed_dict)
            plot_path = os.path.join(args.log_dir, 'plots',
                                        'epoch_%d_step_%d_%s.png' % (epoch, step, ids[0]))
            plot_pcd_three_views(plot_path, all_pcds, model.visualize_titles)
        if epoch % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))

    # tf.reset_default_graph()
    # # pdb.set_trace()
    # save_dir = os.path.join(args.lmdb_train,'stage_'+str(args.stage))
    # if not  os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # else:
    #     shutil.rmtree(save_dir)
    #     os.mkdir(save_dir)

    # is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    # global_step = tf.Variable(0, trainable=False, name='global_step')
    # coarse_pl = tf.placeholder(tf.float32, (1, 1024, 3), 'inputs')
    # features_pl = tf.placeholder(tf.float32, (1, 1024))
    # npts_pl = tf.placeholder(tf.int32, (None), 'num_points')
    # gt_pl = tf.placeholder(tf.float32, (1, 4096, 3), 'ground_truths')

    # model_module = importlib.import_module('.%s' % args.model_type, 'models')
    # model = model_module.Model(features_pl, coarse_pl, gt_pl)

    # if args.lr_decay:
    #     learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
    #                                                args.lr_decay_steps, args.lr_decay_rate,
    #                                                staircase=True, name='lr')
    #     learning_rate = tf.maximum(learning_rate, args.lr_clip)
    # else:
    #     learning_rate = tf.constant(args.base_lr, name='lr')
    # # trainer = tf.train.RMSPropOptimizer(learning_rate)
    # trainer = tf.train.AdamOptimizer(learning_rate)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # sess = tf.Session(config=config)
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))

    # for indx in training_indx:
    #     btch_indx = [indx]
    #     inputs, gt, features, coarse_in, ids = load_training_data_stage3(btch_indx, training_list, args.lmdb_train)
    #     inputs = np.expand_dims(inputs, 0)
    #     features = np.expand_dims(features, 0)
    #     gt = np.expand_dims(gt, 0)
    #     feed_dict = {middle_pl: inputs, features_pl:features, coarse_pl:coarse_in, gt_pl: gt, is_training_pl: False}
    #     middle_out = sess.run([model.middle], feed_dict=feed_dict)
    #     middle_out = middle_out[0]
    #     nme = training_list[indx]
    #     pre_nme = nme.split('.')[0]
    #     feat_nme = pre_nme+'_global'+'.npz'
    #     feat_pth = os.path.join(save_dir, feat_nme)
    #     middle_pth = os.path.join(save_dir, nme)
    #     np.savez(middle_pth, input=middle_out)
    #     np.savez(feat_pth, input=features)

    # # pdb.set_trace()
    # save_dir = os.path.join(args.lmdb_valid,'stage_'+str(args.stage))
    # if not  os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # else:
    #     shutil.rmtree(save_dir)
    #     os.mkdir(save_dir)

    # for indx in valid_indx:
    #     btch_indx = [indx]
    #     inputs, gt, features, ids = load_training_data_stage2(btch_indx, valid_list, args.lmdb_valid)
    #     inputs = np.expand_dims(inputs, 0)
    #     features = np.expand_dims(features, 0)
    #     gt = np.expand_dims(gt, 0)
    #     feed_dict = {coarse_pl: inputs, features_pl:features, gt_pl: gt, is_training_pl: False}
    #     middle_out = sess.run([model.middle], feed_dict=feed_dict)
    #     middle_out = middle_out[0]
    #     nme = valid_list[indx]
    #     pre_nme = nme.split('.')[0]
    #     feat_nme = pre_nme+'_global'+'.npz'
    #     feat_pth = os.path.join(save_dir, feat_nme)
    #     middle_pth = os.path.join(save_dir, nme)
    #     np.savez(middle_pth, input=middle_out)
    #     np.savez(feat_pth, input=features)
    # sess.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/')
    parser.add_argument('--lmdb_valid', default='/shared/kgcoe-research/mil/harish/pcn_data/pcd_datav2/valid/')
    parser.add_argument('--log_dir', default='log/pcn_emd_stage3')
    parser.add_argument('--model_type', default='pcn_emd_stage3')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=1024)
    parser.add_argument('--num_gt_points', type=int, default=4096)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=5)
    parser.add_argument('--steps_per_visu', type=int, default=5)
    parser.add_argument('--steps_per_save', type=int, default=5)
    parser.add_argument('--visu_freq', type=int, default=5)
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