import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, features, coarse, gt):
        self.num_coarse = 1024
        self.grid_size = 2
        self.grid_scale = 0.05
        self.out1 = self.grid_size ** 2 * self.num_coarse
        self.num_fine = self.grid_size ** 2 * self.out1
        self.features = features
        self.coarse = coarse
        self.middle = self.create_decoder(self.features, self.coarse)
        self.loss, self.update = self.create_loss(self.middle, gt)
        self.outputs = self.coarse
        self.visualize_ops = [self.coarse[0], self.middle[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['coarse input', 'middle output', 'ground truth']


    def create_decoder(self, features, coarse):
        
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

                center = mlp_conv(center, [512, 256, 3])
                coarse_fine = mlp_conv(feat, [1024, 512, 256, 3]) + center  #(32, 4096, 3)

            return coarse_fine

   
    def create_loss(self, coarse, gt):
        loss_middle = earth_mover(coarse, gt)
        add_train_summary('train/middle_loss', loss_middle)
        update_middle = add_valid_summary('valid/middle_loss', loss_middle)

        # loss = loss_coarse + alpha * loss_fine
        loss = loss_middle
        # loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_middle, update_loss]