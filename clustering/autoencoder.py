from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


import sys
sys.path.append('../')
from lib.ops import fc, conv2d, deconv2d

class NoisyAutoencoder(object):

    def __init__(self, config):
        self.config = config

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim, self.config.dim, self.config.c],
        )

        # clean #
        tar0 = self.X
        tar1, tar2, self.z_clean, rec2_clean, rec1_clean, rec0_clean = \
                    self.build_enc_dec(self.X, with_drop=False)
        # noisy #
        _, _, self.z_noisy, rec2, rec1, rec0 = \
                    self.build_enc_dec(self.X, with_drop=True)

        loss0 = tf.reduce_mean( tf.square(rec0 - tar0) )
        loss1 = tf.reduce_mean( tf.square(rec1 - tar1) ) * self.config.hlayer_loss_param
        loss2 = tf.reduce_mean( tf.square(rec2 - tar2) ) * self.config.hlayer_loss_param

        loss0_clean = tf.reduce_mean( tf.square(rec0_clean - tar0) )
        loss1_clean = tf.reduce_mean( tf.square(rec1_clean - tar1) ) * self.config.hlayer_loss_param
        loss2_clean = tf.reduce_mean( tf.square(rec2_clean - tar2) ) * self.config.hlayer_loss_param

        self.rX_noisy = rec0
        self.rX_clean = rec0_clean
        self.loss_recons_noisy = loss0 + loss1 + loss2
        self.loss_recons_clean = loss0_clean + loss1_clean + loss2_clean

        self.net_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'enc_dec')

    def build_enc_dec(self, X, with_drop=True, dropout_rate=[0.1, 0.1, 0.], filters=[4, 5], strides=[2,2], name='enc_dec'):

        l_relu = lambda v: tf.nn.leaky_relu(v, alpha=0.01)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            X_c = tf.layers.dropout(X, rate=dropout_rate[0], training=with_drop)

            l_e1 = tf.layers.dropout(
                        conv2d(X_c, 50, activation_fn=l_relu, kernel_size=filters[0], name='l_e1'),
                        rate=dropout_rate[1], training = with_drop 
            )

            l_e2 = tf.layers.dropout(
                        conv2d(l_e1, 50, activation_fn=l_relu, kernel_size=filters[1], name='l_e2'),
                        rate=dropout_rate[2], training = with_drop 
            ) 

            l_e2_flat = tf.contrib.layers.flatten(l_e2)
            l_e3 = fc(l_e2_flat, self.config.z_dim, activation_fn=tf.tanh, name='l_e3')

            l_d2_flat = fc(l_e3, l_e2_flat.get_shape()[1], activation_fn=l_relu, name='l_d2_flat')
            l_d2 = tf.reshape(l_d2_flat, tf.shape(l_e2))

            l_d1 = deconv2d(l_d2, 50, activation_fn=l_relu, kernel_size=filters[1], name='l_d1')
            l_d0 = deconv2d(l_d1, self.config.c, activation_fn=tf.tanh, kernel_size=filters[0], name='l_d0')

            return l_e1, l_e2, l_e3, l_d2, l_d1, l_d0




