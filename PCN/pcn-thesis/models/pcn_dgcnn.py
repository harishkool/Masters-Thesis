
import tensorflow as tf
from tf_util import *
from models.dgcnn_reg import get_model
import pdb

class Model:
    def __init__(self, inputs_1, inputs_2, inputs_3, vox, coarse, middle, gt, is_training, bn_decay, alpha, beta):
        # pdb.set_trace()
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.num_coarse = 1024
        self.grid_size = 2
        self.grid_scale = 0.05
        self.out1 = self.grid_size ** 2 * self.num_coarse
        self.num_fine = self.grid_size ** 2 * self.out1
        self.inputs1 = inputs_1
        self.inputs2 = inputs_2
        self.inputs3 = inputs_3
        self.voxels = vox
        self.coarse_gt = coarse
        self.middle_gt = middle
        self.vox_feat = self.vfe_layer(self.voxels) # bn x 1024
        self.enc_feat = self.create_encoder()
        self.coarse, self.middle, self.fine = self.create_decoder(self.enc_feat, self.vox_feat)
        self.loss, self.update = self.create_loss(self.coarse, self.middle, self.fine, gt, alpha, beta)
        self.outputs = self.fine
        self.visualize_ops = [self.inputs1[0], self.coarse[0], self.middle[0], self.fine[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output','middle output','fine output', 'ground truth']

    def vfe_layer(self, voxels):
        with tf.variable_scope('vfe', reuse=tf.AUTO_REUSE):
            #consider voxels shape is 1 x 50 x 40 x 3
            #mlp_conv -> 1 x 50 x 40 x 256
            #mlp_conv -> 1 x 50 x 40 x 259
            #mlp_conv -> 1 x 50 x 40 x 259 -> 1 x 50 x 40 x 256 -> 1 x 50 x 40 x 512 -> 1 x 50 x 512
            #1 x 50 x 1024
            #return bn x 1024
            with tf.variable_scope('vfe_1', reuse=tf.AUTO_REUSE):
                vox_f = mlp_conv2d(voxels, [128], bn=True, is_training=self.is_training, bn_decay=self.bn_decay)  #--> 1 x 50 x 40 x 128
                max_1 = tf.reduce_max(vox_f, axis=2)  #--> 1 x 50 x 128
                max1_tile = tf.tile(tf.expand_dims(max_1,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 128
                vox_f = tf.concat([voxels, max1_tile], axis=3) #--> 1 x 50 x 40 x 131

            with tf.variable_scope('vfe_2', reuse=tf.AUTO_REUSE):
                vox_f = mlp_conv2d(vox_f, [256], bn=True, is_training=self.is_training, bn_decay=self.bn_decay)  #--> 1 x 50 x 40 x 256
                max_2 = tf.reduce_max(vox_f, axis=2)  #--> 1 x 50 x 256
                max2_tile = tf.tile(tf.expand_dims(max_2,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 256
                vox_f2 = tf.concat([vox_f, max2_tile], axis=3) #--> 1 x 50 x 40 x 512

            with tf.variable_scope('vfe_3', reuse=tf.AUTO_REUSE):
                vox_f = mlp_conv2d(vox_f2, [512], bn=True, is_training=self.is_training, bn_decay=self.bn_decay)  #--> 1 x 50 x 40 x 512
                max_3 = tf.reduce_max(vox_f,axis=2)  #--> 1 x 50 x 512
                max3_tile = tf.tile(tf.expand_dims(max_3,2), [1, 1, tf.shape(vox_f)[2], 1])  #--> 1 x 50 x 40 x 512
                vox_f3 = tf.concat([vox_f, max3_tile], axis=3) #--> 1 x 50 x 40 x 1024

            vox_feat = tf.reduce_max(vox_f3, axis=2)  #--> 1 x 50 x 1024
            vf_features = mlp_conv(vox_feat, [512, 1024], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) #bn x n_v x 1024
            vf_features = tf.reduce_max(vf_features, axis=1) #bn x 1024
        return vf_features


    def create_encoder(self):
        # pdb.set_trace()
        rot1 = tf.squeeze(get_model(self.inputs1, self.is_training, 1, self.bn_decay,)) # bn x 1024
        rot2 = tf.squeeze(get_model(self.inputs2, self.is_training, 2, self.bn_decay)) # bn x 1024
        rot3 = tf.squeeze(get_model(self.inputs3, self.is_training, 3, self.bn_decay)) # bn x 1024
        features = tf.concat([rot1, rot2, rot3, self.vox_feat], axis=1)  # bn x 4096
        features = mlp_conv(tf.expand_dims(features, axis=1), [512, 1024], bn=True, bn_decay=self.bn_decay, is_training=self.is_training)  # bn x 1024
        return features

    def create_decoder(self, features, vox_feat):
        #features --> bn x 1024
        #vox_feat --> bn x 1024
        # pdb.set_trace()
        features = tf.squeeze(features)
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

            vox_tile = tf.tile(tf.expand_dims(vox_feat, 1), [1, self.out1, 1]) # (32, 4096, 1024)
            feat = tf.concat([grid_feat, point_feat, global_feat, vox_tile], axis=2)     #(32, 4096, 2053)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1]) #()
            center = tf.reshape(center, [-1, self.out1, 3])

            coarse_fine = mlp_conv(feat, [512, 512, 3], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) + center  #(32, 4096, 3)

        with tf.variable_scope('folding_2', reuse=tf.AUTO_REUSE):
            grid2 = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size)) #2 x 2
            grid2 = tf.expand_dims(tf.reshape(tf.stack(grid2, axis=2), [-1, 2]), 0) #4 x 2
            grid2_feat = tf.tile(grid2, [features.shape[0], self.out1, 1]) #(32, 16384, 2)

            point2_feat = tf.tile(tf.expand_dims(coarse, 2),[1, 1, self.grid_size ** 4, 1])
            point2_feat = tf.reshape(point2_feat, [-1, self.num_fine, 3]) #(32, 16384, 3)

            fold2_feat = tf.tile(tf.expand_dims(coarse_fine, 2), [1, 1, self.grid_size ** 2, 1])
            fold2_feat = tf.reshape(fold2_feat, [-1, self.num_fine, 3])  #(32, 16384, 3)

            vox_tile2 = tf.tile(tf.expand_dims(vox_feat, 1), [1, self.num_fine, 1]) # (32, 16384, 1024)

            global2_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1]) #(32, 16384, 1024)
            #[32,16384,2], [32,16384,3], [32,16384,3], [?,16384,1024]
            print('global2_feat shape is {}'.format(global2_feat.get_shape()))
            final_feat = tf.concat([grid2_feat, point2_feat, fold2_feat, global2_feat, vox_tile2], axis=2)   #(32, 16384, 2052)

            center2 = tf.tile(tf.expand_dims(coarse_fine, 2), [1, 1, self.grid_size ** 2, 1])
            center2 = tf.reshape(center2, [-1, self.num_fine, 3])

            fine = mlp_conv(final_feat, [512, 512, 3], bn=True, bn_decay=self.bn_decay, is_training=self.is_training) + center2 

        return coarse, coarse_fine, fine

    def create_loss(self, coarse, coarse_fine, fine, gt, alpha, beta):
        print(alpha)
        print(beta)
        loss_coarse = earth_mover(coarse, self.coarse_gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_coarsefine = earth_mover(coarse_fine, self.middle_gt)
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
