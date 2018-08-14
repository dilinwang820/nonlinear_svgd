from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from lib.models import GaussianMixture
from lib.ops import svgd_gradient, fc, conv2d, deconv2d, sqr_dist
from pprint import pprint

from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distribution_util

from autoencoder import NoisyAutoencoder

class SVGD(NoisyAutoencoder):

    def __init__(self, config):
        NoisyAutoencoder.__init__(self, config)

        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None, self.config.num_clusters],
        )

        ''' mixture models '''
        self.phi = tf.ones(self.config.num_clusters) /self.config.num_clusters

        with tf.variable_scope('mixture_model'):
            self.mu = tf.get_variable('mu',
                    shape=(self.config.num_clusters, self.config.z_dim), dtype=tf.float32,
                    initializer = tf.random_uniform_initializer(-0.5, 0.5)) # k x dz x dz


            self.scale_diag = tf.get_variable('scale_diag',
                    shape=(self.config.num_clusters, self.config.z_dim),
                    initializer=tf.random_uniform_initializer(-0.1, 0.1), dtype=tf.float32)

            #scale = []
            #for k in range(self.config.num_clusters):
            #    scale.append( tf.contrib.distributions.fill_triangular(self.scale_tril[k]) )
            #self.scale = tf.stack(scale, axis=0)
            self.scale = tf.matrix_diag(self.scale_diag )
            self.scale = 0*self.scale + tf.matrix_diag( tf.ones((self.config.num_clusters, self.config.z_dim)) ) #discard covaraince

            self.cov = tf.matmul(self.scale, self.scale, transpose_b=True) # k x dz x dz
            assert self.cov.get_shape() == [self.config.num_clusters, self.config.z_dim, self.config.z_dim], 'illegal dim cov'
            #self.cov = tf.exp( self.scale_tril )

        self.gmm_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mixture_model')

        self.energy_noisy, logits_noisy = self.compute_energy(self.z_noisy, self.phi, self.mu, self.cov) # TODO
        self.energy_clean, logits_clean = self.compute_energy(self.z_clean, self.phi, self.mu, self.cov) # TODO

        self.z = self.z_clean
        self.pred_noisy = tf.nn.softmax(logits_noisy, -1)
        self.pred_clean = tf.nn.softmax(logits_clean, -1)

        self.loss_clus = tf.reduce_mean( tf.losses.softmax_cross_entropy(self.y, logits_noisy) )

        self.depict_loss = self.loss_recons_noisy + self.loss_clus 
        self.loss = self.energy_noisy

        # compute gradients
        self.train_grads, self.train_vars = [], []
        self.train_vars += self.net_train_vars
        self.net_grads = tf.gradients(self.loss, self.net_train_vars)
        self.train_grads += self.net_grads

        self.gmm_train_grads = []
        for var in self.gmm_train_vars:
            grad = tf.gradients(self.loss, var)[0] # return a list
            pprint(grad)
            #app_gmm_grad.append(vg)
            svgd_grad = svgd_gradient(var, grad, kernel=self.config.kernel, temperature=self.config.temperature)
            self.gmm_train_grads.append(svgd_grad)

        self.train_vars += self.gmm_train_vars
        self.train_grads += self.gmm_train_grads

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("recon_err", self.loss_recons_noisy)
        tf.summary.scalar("loss_clus", self.loss_clus)
        tf.summary.scalar("energy_noisy", self.energy_noisy)
        tf.summary.scalar("energy_clean", self.energy_clean)

    def get_feed_dict(self, X, y=None):
        fd = {
            self.X: X,  
        }
        if y is not None:
            fd[self.y] = y
        return fd


    def compute_energy(self, z, phi, mu, scale):

        def _sum_log_exp(z, weights, mu, scale):
            diff = tf.expand_dims(z, 0) - tf.expand_dims(mu, 1)  # c x n x d
            ##diff_times_inv_cov = diff * tf.expand_dims(1./ cov, 1)  # c x n x d
            #diff_times_inv_cov = tf.reduce_sum(tf.expand_dims(diff, 3) * \
            #            tf.expand_dims(tf.matrix_inverse(cov), 1), axis=-2) # c x n x d = (c x n x d x 1) * (c x 1 x d x d)

            #sum_sq_dist_times_inv_cov = tf.reduce_sum(diff_times_inv_cov * diff, axis=-1)

            # https://www.tensorflow.org/api_docs/python/tf/contrib/distributions/MultivariateNormalTriL
            scale_inv_trans = tf.transpose(tf.matrix_inverse(scale), [0, 2, 1])
            y = tf.reduce_sum(tf.expand_dims(diff, 3) * tf.expand_dims(scale_inv_trans, 1), axis=-2) # c x n x d
            sum_sq_dist_times_inv_cov = tf.reduce_sum(tf.square(y), axis=-1)

            ln2piD = tf.log(2 * np.pi) * self.config.z_dim

            #lnD = tf.reduce_sum(tf.log(cov), axis=1)
            #lnD = tf.squeeze( tf.linalg.logdet(cov) ) # c
            lnD = tf.squeeze(2.0*tf.reduce_sum(tf.log(( tf.sqrt(1e-8 + tf.matrix_diag_part(scale)**2) )), axis=-1))

            log_coefficients = tf.expand_dims(ln2piD + lnD, 1) # c x 1
            log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
            log_weighted = log_components + tf.expand_dims(tf.log(weights), 1)  # c x n + c x 1
            log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
            return log_weighted, log_shift


        log_weighted, log_shift = _sum_log_exp(z, phi, mu, scale)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, axis=0, keep_dims=True)
        logp = tf.log(exp_log_shifted_sum) + log_shift # 1 x n

        energy = -logp
        logits = tf.transpose(log_shift + log_weighted, [1, 0])
        return tf.reduce_mean(tf.maximum(1., energy)), logits


