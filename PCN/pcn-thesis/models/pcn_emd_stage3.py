import tensorflow as tf
from tf_util import *


class Model:
    def __init__(self, features, coarse, middle, gt):
        self.num_coarse = 1024
        self.grid_size = 2
        self.grid_scale = 0.05
        self.out1 = self.grid_size ** 2 * self.num_coarse
        self.num_fine = self.grid_size ** 2 * self.out1
        self.features = features
        self.coarse = coarse
        self.middle = middle
        self.fine = self.create_decoder(self.features, self.coarse, self.middle)
        self.loss, self.update = self.create_loss(self.fine, gt)
        self.visualize_ops = [self.coarse[0], self.middle[0], self.fine[0], gt[0]]
        # self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, gt]
        self.visualize_titles = ['coarse input', 'middle output', 'fine output', 'ground truth']


    def create_decoder(self, features, coarse, coarse_fine):
        
            with tf.variable_scope('folding_2', reuse=tf.AUTO_REUSE):
                grid2 = tf.meshgrid(tf.linspace(-0.05, 0.05, self.grid_size), tf.linspace(-0.05, 0.05, self.grid_size)) #2 x 2
                grid2 = tf.expand_dims(tf.reshape(tf.stack(grid2, axis=2), [-1, 2]), 0) #4 x 2
                grid2_feat = tf.tile(grid2, [features.shape[0], self.out1, 1]) #(32, 16384, 2)

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

            return fine

   
    def create_loss(self, fine, gt):
        # loss_fine = chamfer(fine, gt)
        loss_fine = earth_mover(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        # loss = loss_coarse + alpha * loss_fine
        loss = loss_fine
        # loss = loss_fine
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_fine, update_loss]