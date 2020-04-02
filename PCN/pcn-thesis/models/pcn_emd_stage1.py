import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, inputs, gt,voxels):
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
        self.coarse = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, gt)
        self.outputs = self.coarse
        self.visualize_ops = [inputs[0], self.coarse[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'ground truth']

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
    
        return coarse

   
    def create_loss(self, coarse, gt):
        loss_coarse = earth_mover(coarse, gt)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        # loss = loss_coarse + alpha * loss_fine
        loss = loss_coarse
        # loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_loss]