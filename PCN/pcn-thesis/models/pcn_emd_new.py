import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, gt, voxels, coarse, alpha, beta, is_training=None, bn_decay=None):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.bn_decay = bn_decay
        self.coarse_gt = coarse
        self.is_training = is_training
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.vox_features = self.vfe_layer(voxels)
        self.features = self.create_encoder(inputs, self.vox_features)
        self.features = tf.reshape(self.features,[-1,self.num_coarse])
        # self.features = tf.reshape(self.features, [tf.shape(self.features)[0], tf.shape(self.features)[2]])
        # print('features after encoder shape is {}'.format(self.features.get_shape()))
        self.coarse, self.fine = self.create_decoder(self.features, self.vox_features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha, beta)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output','fine output', 'ground truth']

    def vfe_layer(self, voxels):
            #consider voxels shape is 1 x 50 x 40 x 3
            #mlp_conv -> 1 x 50 x 40 x 256
            #mlp_conv -> 1 x 50 x 40 x 259
            #mlp_conv -> 1 x 50 x 40 x 259 -> 1 x 50 x 40 x 256 -> 1 x 50 x 40 x 512 -> 1 x 50 x 512
            #1 x 50 x 1024
            #return bn x 1024
        with tf.variable_scope('vfe_1', reuse=tf.AUTO_REUSE):
            vox_f = mlp_conv2d_reg(voxels, [128], bn=True, bn_decay=self.bn_decay, is_training=self.is_training)  #--> 1 x 50 x 40 x 128
            max_1 = tf.reduce_max(vox_f, axis=2)  #--> 1 x 50 x 128
            max1_tile = tf.tile(tf.expand_dims(max_1,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 128
            vox_f = tf.concat([voxels, max1_tile], axis=3) #--> 1 x 50 x 40 x 131

        with tf.variable_scope('vfe_2', reuse=tf.AUTO_REUSE):
            vox_f = mlp_conv2d_reg(vox_f, [256], bn=True, bn_decay=self.bn_decay, is_training=self.is_training)  #--> 1 x 50 x 40 x 256
            max_2 = tf.reduce_max(vox_f, axis=2)  #--> 1 x 50 x 256
            max2_tile = tf.tile(tf.expand_dims(max_2, 2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 256
            vox_f2 = tf.concat([vox_f, max2_tile], axis=3) #--> 1 x 50 x 40 x 512

        with tf.variable_scope('vfe_3', reuse=tf.AUTO_REUSE):
            vox_f = mlp_conv2d_reg(vox_f2, [512], bn=True, bn_decay=self.bn_decay, is_training=self.is_training)  #--> 1 x 50 x 40 x 512
            max_3 = tf.reduce_max(vox_f, axis=2)  #--> 1 x 50 x 512
            max3_tile = tf.tile(tf.expand_dims(max_3, 2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 512
            vox_f3 = tf.concat([vox_f, max3_tile], axis=3) #--> 1 x 50 x 40 x 1024

        vox_feat = tf.reduce_max(vox_f3, axis=2)  #--> 1 x 50 x 1024
        vf_features = mlp_conv_reg(vox_feat, [512, 1024], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) #bn x n_v x 1024
        vf_features = tf.reduce_max(vf_features, axis=1) #bn x 1024
        return vf_features

    def create_encoder(self, inputs, vox_features):
        #inputs --> bn x n x 3
        #vox_features --> bn x n_v x v_featr  
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv_reg(inputs, [128, 256], bn=True, bn_decay=self.bn_decay, is_training=self.is_training)  # bn x N x 256
            features_global = tf.reduce_max(features, axis=1, name='maxpool_0') # bn x 256
            # print('features global shape is {}'.format(features_global.get_shape()))
            features_global2 = tf.tile(tf.expand_dims(features_global, 1), [1, tf.shape(features)[1], 1]) 
            features = tf.concat([features, features_global2], axis=2) #bn x N x 512
            features = mlp_conv_reg(features, [512, 1024], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) #bn x n x 1024
            features = tf.reduce_max(features, axis=1, name='maxpool_1') #bn x 1024

        with tf.variable_scope('vox_pw_concat', reuse=tf.AUTO_REUSE):
            features = tf.concat([features, vox_features], axis=1) #bn x 2048
            # features = mlp_conv2d(features, [512, 1024])  # bn x 1024
            features = mlp(features, [2048, 1024])
        return features

    def create_decoder(self, features, vox_feat):
        with tf.variable_scope('coarse', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])  #(32, 1024, 3) 


        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            vox_tile2 = tf.tile(tf.expand_dims(vox_feat, 1), [1, self.num_fine, 1]) # (32, 16384, 1024)

            feat = tf.concat([grid_feat, point_feat, global_feat, vox_tile2], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv_reg(feat, [512, 512, 3], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) + center

        return coarse, fine

   
    def create_loss(self, coarse, fine, gt, alpha, beta):
        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, self.coarse_gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        # loss_coarsefine = chamfer(coarse_fine, gt)
        # gt_coarsefine = gt[:, :coarse_fine.shape[1], :]
        # loss_coarsefine = earth_mover(coarse_fine, self.middle_gt)
        # add_train_summary('train/middle_loss', loss_coarsefine)
        # update_coarsefine = add_valid_summary('valid/middle_loss', loss_coarsefine)
        
        loss_fine = chamfer(fine, gt)
        # loss_fine = earth_mover(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        # loss = loss_coarse + alpha * loss_fine
        # loss = loss_coarse + alpha*loss_coarsefine + beta*loss_fine
        # loss = loss_coarse + alpha*loss_fine
        loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_loss]