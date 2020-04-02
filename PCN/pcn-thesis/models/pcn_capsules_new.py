# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from tf_util import *
import numpy as np

class Model:
    def __init__(self, inputs, gt, alpha, beta, args):
        self.args = args
        self.num_coarse = 1024
        self.grid_size = 2
        self.grid_scale = 0.05
        self.capsule_out_shape = 2048
        self.primary_capsules = 16
        self.latent_cap_size = 64
        self.latent_cap_num = 64
        self.num_iterations = 3
        self.primary_out = self.primary_capsules*self.num_coarse
        self.num_primitives = 16
        self.out1 = self.grid_size ** 2 * self.num_coarse
        self.num_fine = self.grid_size ** 2 * self.out1
        self.features = self.create_encoder(inputs)
        print(self.features.get_shape())
        self.features = tf.reshape(self.features, [-1, tf.shape(self.features)[1], tf.shape(self.features)[3]])
        self.coarse, self.fine = self.create_decoder(self.features)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, gt, alpha)
        self.outputs = self.fine
        self.visualize_ops = [inputs[0], self.coarse[0], self.fine[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'ground truth']

    #shud I use primary capsules directly or use pcn style global vector in the folding_2 decoder????
    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])  # bn x N x 256
            features_global = tf.reduce_max(features, axis=1, name='maxpool_0') # bn x 256
            # print('features global shape is {}'.format(features_global.get_shape()))
            features_global2 = tf.tile(tf.expand_dims(features_global, 1), [1, tf.shape(features)[1], 1]) 
            features = tf.concat([features, features_global2], axis=2) #bn x N x 512
            features = mlp_conv(features, [512, self.primary_out])
            features = tf.reshape(features, [-1, self.primary_capsules, tf.shape(features)[1], self.num_coarse])
            #like above we need bn x no.primary capsules  x n x 1024
            features = tf.reduce_max(features, axis=2, keep_dims=True, name='maxpool_1') #bn x num_primarycapsules x 1024
            features = self.squash_function(features)
        return features

    def squash_function(self, features):
        squared_norm = tf.reduce_sum(tf.square(features),-1, keepdims=True)
        scalar_factor = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm)
        out_squashes = scalar_factor*features
        return out_squashes
    
    def create_decoder(self, features):
        #features shape is bn x primary capsules x 1024 e.g., 1 x 16 x 1024
        #make it into 1 x 1024 x 16
        # latent capsules -> dim -> 64
        # num latent capsules -> 64
        # 16 x 64 --> weight matrix
        # 16 x 64 --> has to happen for all 64 latent capsules, so 16 x 64 x 64
        # total 1024 primary capsules, total weight matrix will be 1024 x 16 x 64 x 64 
        # bij is primary capsules coupled to latent capsules
        # bij is 1 x 1024 x 64
        # bn x 1024 x 64x 64 
        # weight is 4 x 1024 x 16 x 64 x 64 --> expand dims --> 4 x 1024 x 64 x 64 x 16 x 1
        # inpt is 4 x 1024x 16 --> tile 64 x 64 ---> 4 x 1024 x 16 x 64 x 64 --> expan_dims 4 x 1024 x 64 x     64 x 1x 16
        # output shud be 4 x 1024 x 64 x 64
        global_features = features
        features_shape = get_shape(features)
        weights = tf.random_normal_initializer()
        with tf.variable_scope('latent_capsules', reuse=tf.AUTO_REUSE):
            W = tf.get_variable('Weight', shape=[self.args.batch_size, self.num_coarse, 
                                self.latent_cap_size, self.latent_cap_num, 1, self.primary_capsules],
                        dtype=tf.float32, initializer=weights)
            #weights shape is bn x 1024 x 16 x 64 x 64
            # biases = tf.get_variable('bias', shape=(1, 1, self.latent_cap_size, self.latent_cap_num, 1))
            features = tf.tile(features, [1,self.latent_cap_num, self.latent_cap_size])
            features = tf.reshape(features, [-1, self.num_coarse, self.latent_cap_size, self.latent_cap_num,
                        self.primary_capsules, 1])
            latent_caps = tf.matmul(W, features)  # bn x 1024 x 64 x 64
            b_ij = tf.constant(np.zeros((self.args.batch_size, self.num_coarse, self.latent_cap_size)),dtype=tf.float32)
        latent_caps_stop = tf.stop_gradient(latent_caps, name='stop_gradient')
        latent_caps_stop = tf.reshape(latent_caps_stop, [-1, tf.shape(latent_caps_stop)[1], tf.shape(latent_caps_stop)[2], 
                tf.shape(latent_caps_stop)[3]])
        latent_caps = tf.reshape(latent_caps, [-1, tf.shape(latent_caps)[1], tf.shape(latent_caps)[2], 
                tf.shape(latent_caps)[3]])
        with tf.variable_scope('routing', reuse=tf.AUTO_REUSE):
            c_ij = tf.nn.softmax(b_ij, axis=2)
            for i in range(self.num_iterations):
                if i==self.num_iterations-1:
                    c_ij = tf.tile(tf.expand_dims(c_ij,3),[1,1,1,self.latent_cap_size])
                    s_ij = tf.multiply(latent_caps, c_ij)  # s_ij shud be bn x 1024 x 64 x 64
                    v_j = self.squash_function(tf.reduce_sum(s_ij,axis=1))  #v_j should be bn x 64 x 64
                else:
                    #latent_caps_stop bn x 64 x 64
                    #c_ij bn x 1024 x 64 
                    c_ij = tf.tile(tf.expand_dims(c_ij,3),[1,1,1,self.latent_cap_size])
                    s_ij = tf.multiply(latent_caps_stop, c_ij)
                    #v_j bn x 64 x 64???
                    v_j = self.squash_function(tf.reduce_sum(s_ij,axis=1))
                    #b_ij bn x 1024 x 64, v_j is 64 x 64, latent_caps_stop bn x 1024 x 64 x 64
                    #output of multiply is bn x 1024 x 64 x 64
                    v_j_tile = tf.tile(tf.expand_dims(v_j, axis=1), [1, self.num_coarse, 1,1])
                    b_ij = b_ij + tf.reduce_sum(tf.multiply(latent_caps_stop, v_j_tile), axis=-1)
                    c_ij = tf.nn.softmax(b_ij, axis=2)

        with tf.variable_scope('folding_1', reuse=tf.AUTO_REUSE):
            # random_grid = tf.get_variable('random_grid',shape=[tf.shape(features)[0], 16, 2],
            #             dtype = tf.float32, initializer =tf.random_normal_initializer())
            # random_grid = tf.tile(random_grid,[1, self.num_primitives])
            # v_j = tf.tile(v_j,[1,1,self.num_primitives])
            # concat_feat = tf.concat([random_grid,v_j], axis=2)
            coarse = conv_layers(v_j, self.num_primitives, [64, 64, 32, 16, 3], self.args)  #output size is bn x 1024 x 3

        print('coarse shape is {}'.format(coarse.get_shape()))
        # with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):         
        #     coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
        #     coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])  #(32, 1024, 3) 
        # 2048 x 4 = 8192
        with tf.variable_scope('folding_2', reuse=tf.AUTO_REUSE):
            # grid = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size*2), tf.linspace(-0.05, 0.05, self.grid_size*2)) #2 x 2
            # grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0) #16 x 2 
            grid = tf.get_variable(
                "grid1", [1, 16, 2], initializer=tf.random_uniform_initializer(0, 1)
            )
            grid_feat = tf.tile(grid, [coarse.shape[0], coarse.shape[1], 1])   #(bn, 16384, 2)

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size **4, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])         #(32, 16384, 3)
            global_features = tf.reduce_max(global_features, axis=1)  #global_features is bn x 16 x 1024 --> reduce_max to bn x 1024
            # print('global feat shape {}'.format(global_features.get_shape()))
            global_feat = tf.tile(tf.expand_dims(global_features,axis=1), [1, self.num_fine, 1])  #
            
            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)     #(32, 16384, 1029)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 4, 1]) #()
            center = tf.reshape(center, [-1, self.num_fine, 3])

            # coarse_fine = mlp_conv(feat, [512, 512, 3]) + center  #(32, 4096, 3)
            feat = tf.reshape(feat, [-1, self.num_fine, 1029])
            fine = mlp_conv(feat, [512, 512, 3]) + center

        return coarse, fine

    def create_loss(self, coarse, fine, gt, alpha):
        gt_ds2 = gt[:, :coarse.shape[1], :]
        # loss_coarse = earth_mover(coarse, gt_ds2)
        # loss_coarse = chamfer(coarse, gt_ds2)
        # add_train_summary('train/coarsefine_loss', loss_coarse)
        # update_coarse = add_valid_summary('valid/coarsefine_loss', loss_coarse)
        
        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        # loss = loss_coarse + alpha * loss_fine
        # loss = loss_coarse + alpha*loss_fine
        loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss) 

        # return loss, [update_coarse, update_fine, update_loss]
        return loss, [update_loss]


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