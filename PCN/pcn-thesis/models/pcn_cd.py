# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, gt,voxels, alpha, beta):
        self.num_coarse = 1024
        self.grid_size = 2
        self.grid_scale = 0.05
        self.out1 = self.grid_size ** 2 * self.num_coarse
        self.num_fine = self.grid_size ** 2 * self.out1
        self.vox_features = self.vfe_layer(voxels)
        self.features = self.create_encoder(inputs, self.vox_features)
        self.features = tf.reshape(self.features,[-1,self.num_coarse])
        # self.features = tf.reshape(self.features, [tf.shape(self.features)[0], tf.shape(self.features)[2]])
        # print('features after encoder shape is {}'.format(self.features.get_shape()))
        self.coarse, self.coarse_fine, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, self.coarse_fine, self.fine, gt, alpha, beta)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.coarse_fine[0], self.fine[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output','coarse_fine_output','fine output', 'ground truth']

    def vfe_layer(self, voxels):
        with tf.variable_scope('vfe', reuse=tf.AUTO_REUSE):
            #consider voxels shape is 1 x 50 x 40 x 3
            #mlp_conv -> 1 x 50 x 40 x 256
            #mlp_conv -> 1 x 50 x 40 x 259
            #mlp_conv -> 1 x 50 x 40 x 259 -> 1 x 50 x 40 x 256 -> 1 x 50 x 40 x 512 -> 1 x 50 x 512
            #return 1 x 50 x 512
            with tf.variable_scope('vfe_1', reuse=tf.AUTO_REUSE):
                vox_f = mlp_conv2d(voxels, [128])  #--> 1 x 50 x 40 x 128
                max_1 = tf.reduce_max(vox_f,axis=2)  #--> 1 x 50 x 128
                max1_tile = tf.tile(tf.expand_dims(max_1,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 128
                vox_f = tf.concat([voxels, max1_tile], axis=3) #--> 1 x 50 x 40 x 131

            with tf.variable_scope('vfe_2', reuse=tf.AUTO_REUSE):
                vox_f = mlp_conv2d(vox_f, [256])  #--> 1 x 50 x 40 x 256
                max_2 = tf.reduce_max(vox_f,axis=2)  #--> 1 x 50 x 256
                max2_tile = tf.tile(tf.expand_dims(max_2,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 256
                vox_f2 = tf.concat([vox_f, max2_tile], axis=3) #--> 1 x 50 x 40 x 512

            vox_feat = tf.reduce_max(vox_f2, axis=2)  #--> 1 x 50 x 512
        return vox_feat

    def create_encoder(self, inputs, vox_features):
        #inputs --> bn x n x 3
        #vox_features --> bn x n_v x v_featr
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])  # bn x N x 256
            features_global = tf.reduce_max(features, axis=1, name='maxpool_0') # bn x 256
            # print('features global shape is {}'.format(features_global.get_shape()))
            features_global2 = tf.tile(tf.expand_dims(features_global, 1), [1, tf.shape(features)[1], 1]) 
            features = tf.concat([features, features_global2], axis=2) #bn x N x 512
            features = mlp_conv(features, [512, 1024]) #bn x n x 1024
            features = tf.reduce_max(features, axis=1, name='maxpool_1') #bn x 1024
        
        with tf.variable_scope('encoder_vfe', reuse=tf.AUTO_REUSE):
            vf_features = mlp_conv(vox_features, [128, 256])  # bn x n_v x 256
            #here concatenating point global features, you can concat with global voxel features instead
            vf_features_global2 = tf.tile(tf.expand_dims(features_global, 1), [1, tf.shape(vox_features)[1], 1]) 
            vf_features = tf.concat([vf_features, vf_features_global2], axis=2) #bn x n_v x 512
            vf_features = mlp_conv(vf_features, [512, 1024]) #bn x n_v x 1024
            vf_features = tf.reduce_max(vf_features, axis=1, name='vf_maxpool_1') #bn x 1024

        with tf.variable_scope('vox_pw_concat', reuse=tf.AUTO_REUSE):
            features = tf.concat([features, vf_features], axis=1) #bn x 2048
            # features = mlp_conv2d(features, [512, 1024])  # bn x 1024
            features = mlp(features, [2048, 1024])
        return features


    def create_decoder(self, features):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])  #(32, 1024, 3) 

        with tf.variable_scope('folding_1', reuse=tf.AUTO_REUSE):
            grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size)) #2 x 2
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0) #4 x 2 
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])   #(32, 4096, 2)

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.out1, 3])         #(32, 4096, 3)

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.out1, 1])  #(32, 4096, 1024)

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)     #(32, 4096, 1029)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1]) #()
            center = tf.reshape(center, [-1, self.out1, 3])

            coarse_fine = mlp_conv(feat, [512, 512, 3]) + center  #(32, 4096, 3)

        with tf.variable_scope('folding_2', reuse=tf.AUTO_REUSE):
            grid2_feat = tf.tile(grid, [features.shape[0], self.out1, 1]) #(32, 16384, 2)

            point2_feat = tf.tile(tf.expand_dims(coarse, 2),[1, 1, self.grid_size ** 4, 1])
            point2_feat = tf.reshape(point2_feat, [-1, self.num_fine, 3]) #(32, 16384, 3)

            fold2_feat = tf.tile(tf.expand_dims(coarse_fine, 2), [1, 1, self.grid_size ** 2, 1])
            fold2_feat = tf.reshape(fold2_feat, [-1, self.num_fine, 3])  #(32, 16384, 3)

            global2_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1]) #(32, 16384, 1024)
            #[32,16384,2], [32,16384,3], [32,16384,3], [?,16384,1024]
            print('global2_feat shape is {}'.format(global2_feat.get_shape()))
            final_feat = tf.concat([grid2_feat, point2_feat, fold2_feat, global2_feat], axis=2)   #(32, 16384, 1032)

            center2 = tf.tile(tf.expand_dims(coarse_fine, 2), [1, 1, self.grid_size ** 2, 1])
            center2 = tf.reshape(center2, [-1, self.num_fine, 3])

            fine = mlp_conv(final_feat, [512, 512, 3]) + center2 

        return coarse, coarse_fine, fine

    def create_loss(self, coarse, coarse_fine, fine, gt, alpha, beta):
        loss_coarse = chamfer(coarse, gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_coarsefine = chamfer(coarse_fine, gt)
        add_train_summary('train/middle_loss', loss_coarsefine)
        update_coarsefine = add_valid_summary('valid/middle_loss', loss_coarsefine)
        
        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        # loss = loss_coarse + alpha * loss_fine
        loss = loss_coarse + alpha*loss_coarsefine + beta*loss_fine
        # loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_coarsefine, update_fine, update_loss]


#####  Previous shapes ##########
# GT shape is (32, 16384, 3)
# Features shape is (32, 2048, 256)
# Input shape is (32, 2048, 3)??
# Features global shape is (32, 1, 256)
# Features after concat shape is (32, 2048, 512)
# Final features shape is (32, 1024)
# Grid feat shape is (32, 16384, 2)
# Point feat shape is (32, 16384, 3)
# Global  feat shape is (32, 16384, 1024)
# feat shape is (32, 16384, 1029)