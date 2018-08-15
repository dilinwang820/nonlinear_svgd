from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from lib.models import GaussianMixture
from lib.ops import svgd_gradient, rbf_kernel


class Model(object):

    def __init__(self, config):
        self.config = config

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        ## basically, mus and log_vars; bad initlization
        self.mus = tf.get_variable('mus',
                shape=(self.config.n_components, self.config.dim), dtype=tf.float32,
                initializer = tf.random_uniform_initializer(-self.config.n_components, self.config.n_components))

        self.log_vars = tf.get_variable('log_vars',
                shape=(self.config.n_components, self.config.dim),
                initializer=tf.zeros_initializer(), dtype=tf.float32)

        self.particles = tf.concat((self.mus, self.log_vars), axis=1)

        self.approx = GaussianMixture(self.config.n_components, self.mus, self.log_vars, is_train=False)
        self.log_prob = self.approx.log_prob( self.X)
        mu_grad, log_var_grad = tf.gradients(tf.reduce_mean(self.log_prob), [self.mus, self.log_vars])
        grad = tf.concat((mu_grad, log_var_grad), axis=1)

        svgd_grad = svgd_gradient(self.particles, grad=grad, \
                temperature=self.config.temperature, kernel=self.config.kernel)

        self.clip_op = self.log_vars.assign( tf.clip_by_value(self.log_vars, -3., 3.) )
        updates = -svgd_grad  # maximize log likelihood
        self.train_vars = [self.mus, self.log_vars]
        self.train_grads = [tf.reshape(updates[:, :self.config.dim], self.mus.get_shape()), tf.reshape(updates[:, self.config.dim:], self.log_vars.get_shape())]


    def get_feed_dict(self, X):
        fd = {
            self.X: X,  
        }
        return fd


