from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

from lib.models import GaussianMixture
from lib.ops import svgd_gradient, fc
from pprint import pprint


class SVGD(object):

    def __init__(self, config):
        self.config = config

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        self.h, self.rX = self.ae_network(self.X, name='da')
        self.r_euc_dist = self.relative_euc_dist(self.X, self.rX)
        self.r_cos_dist = self.cosine_similarity(self.X, self.rX)

        self.gamma = tf.ones((tf.shape(self.X)[0], self.config.n_components)) /self.config.n_components
        self.z = tf.concat((tf.reshape(self.h, (-1, 1)), tf.reshape(self.r_euc_dist, (-1, 1)), tf.reshape(self.r_cos_dist, (-1, 1))), axis=1) # n x dz

        self.phi = tf.ones(self.config.n_components) /self.config.n_components

        with tf.variable_scope('mixture_model'):
            self.mu = tf.get_variable('mu',
                    shape=(self.config.n_components, self.config.z_dim), dtype=tf.float32,
                    initializer = tf.random_uniform_initializer(-0.5, 0.5)) # k x dz x dz

            self.scale_tril = tf.get_variable('scale_tril',
                    shape=(self.config.n_components, self.config.z_dim * (self.config.z_dim+1) //2),
                    initializer=tf.random_uniform_initializer(-1., 1.), dtype=tf.float32)

            scale = []
            for k in range(self.config.n_components):
                scale.append( tf.contrib.distributions.fill_triangular(self.scale_tril[k]) )
            self.scale = tf.stack(scale, axis=0)

            self.cov = tf.matmul(self.scale, self.scale, transpose_b=True) # k x dz x dz
            assert self.cov.get_shape() == [self.config.n_components, self.config.z_dim, self.config.z_dim], 'illegal dim cov'

        self.net_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'da')
        self.gmm_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mixture_model')

        #print(self.net_train_vars, self.gmm_train_vars)

        # loss
        self.energy = self.compute_energy(self.z, self.phi, self.mu, self.scale)

        self.cov_inv = tf.reduce_sum( 1. /tf.matrix_diag_part(self.cov) )
        self.recon_err = tf.reduce_mean( tf.reduce_sum((self.X - self.rX)**2, axis=1) )
        #self.recon_err = tf.reduce_mean( (self.X - self.rX)**2 )
        self.loss = self.recon_err + self.config.lambda_energy * tf.reduce_mean(self.energy) + self.config.lambda_cov_inv * self.cov_inv 

        self.train_grads, self.train_vars = [], []
        self.train_grads += tf.gradients(self.loss, self.net_train_vars)
        self.train_vars += self.net_train_vars

        app_gmm_grad = []
        for var in self.gmm_train_vars:
            vg = tf.gradients(self.loss, var)[0] # return a list
            app_gmm_grad.append(vg)

        app_gmm_var = tf.concat((self.mu, self.scale_tril), axis=1)
        app_gmm_grad = tf.concat(app_gmm_grad, axis=1)

        svgd_grad = svgd_gradient(app_gmm_var, app_gmm_grad, kernel=self.config.kernel, temperature=self.config.temperature)
        self.train_grads += [svgd_grad[:,:self.config.z_dim], svgd_grad[:, self.config.z_dim:]]
        self.train_vars += self.gmm_train_vars

        #self.clip_op = self.scale_tril.assign( tf.clip_by_value(self.scale_tril, -1., 1.) )

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("recon_err", self.recon_err)
        tf.summary.scalar("cov_inv", self.cov_inv)
        tf.summary.scalar("energy", tf.reduce_mean(self.energy))


    def get_feed_dict(self, X):
        fd = {
            self.X: X,  
        }
        return fd


    def ae_network(self, X, name):
        with tf.variable_scope(name):
            h1 = fc(X, 60, activation_fn=tf.tanh, name='h1')
            h2 = fc(h1, 30, activation_fn=tf.tanh, name='h2')
            h3 = fc(h2, 10, activation_fn=tf.tanh, name='h3')
            fl = fc(h3, 1, activation_fn=None, name='fl')

            l1 = fc(fl, 10, activation_fn=tf.tanh, name='l1')
            l2 = fc(l1, 30, activation_fn=tf.tanh, name='l2')
            l3 = fc(l2, 60, activation_fn=tf.tanh, name='l3')
            rx = fc(l3, self.config.dim, activation_fn=None, name='rx')

            return fl, rx


    def compute_energy(self, z, phi, mu, scale):

        def _sum_log_exp(z, weights, mu, scale):
            #eps = tf.matrix_diag( tf.ones((self.config.n_components, self.config.z_dim)) )
            #cov = tf.matmul(scale, scale, transpose_b=True) + 1e-6 * eps

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
        return tf.squeeze(energy)


    def relative_euc_dist(self, a, b):
        return tf.norm(a - b, axis=1) / tf.norm(a, axis=1)


    def cosine_similarity(self, a, b):
        an = tf.nn.l2_normalize(a, axis=1)
        bn = tf.nn.l2_normalize(b, axis=1)
        return tf.reduce_sum(an * bn, axis=1)





