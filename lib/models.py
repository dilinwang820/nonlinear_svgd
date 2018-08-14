from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import util as distribution_util

import sys
import numpy as np
from math import pi
import pprint
from util import log

def sample_gumbel(n_samples, c, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform([n_samples, c], minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, n_samples, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(n_samples, logits.get_shape()[0].value)
    return tf.nn.softmax( y / temperature, 1)


def gumbel_softmax(logits, n_samples, temperature=0.1, hard=False):
    assert logits.get_shape().ndims == 1, 'illegal inputs'
    """
        Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
            logits: [batch_size, n_class] unnormalized log-probs
            temperature: non-negative scalar
            hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
            [batch_size, n_class] sample from the Gumbel-Softmax distribution.
            If hard=True, then the returned sample will be one-hot, otherwise it will
            be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, int(n_samples), temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


class NormFlow():
    # 1) calculate u_hat to ensure invertibility (appendix A.1 to)
    # 2) calculate the forward transformation of the input f(z) (Eq. 8)
    # 3) calculate u_hat^T psi(z) 
    # 4) calculate logdet-jacobian log|1 + u_hat^T psi(z)| to be used in the LL function
    # 5) z is (batch_size, num_latent), u is (1, num_latent), w is (num_latent, 1), b is (1,)

    def __init__(self, n_layers=10, dim=10):
        log.infov('normalization flow as proposer')
        self.n_layers = n_layers
        self.dim = dim
        self.fu = tf.get_variable('fu', shape=(n_layers, self.dim), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.2))

        self.fw = tf.get_variable('fw', shape=(n_layers, self.dim), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.2))

        self.fb = tf.get_variable('fb', shape=(n_layers,), dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.2))
        self.params = [self.fu, self.fw, self.fb]

    def sample(self, n_samples=128, with_init_noise=False):
        z = tf.random_normal([int(n_samples), self.dim])
        x, _ = self.flow(z)
        if with_init_noise:
            return x, z
        return x


    def log_prob(self, z):
        _, sum_logdet = self.flow(z)
        return -tf.reduce_sum(z**2, axis=1) - sum_logdet


    def flow(self, z):
        sum_logdet = tf.constant(0.)
        for i in range(self.n_layers):
            u = tf.expand_dims(self.fu[i], 0) # 1 x d
            w = tf.expand_dims(self.fw[i], 1) # d x 1
            b = self.fb[i] # 1

            uw = tf.matmul(u, w)  # 1 x 1
            muw = -1. + tf.nn.softplus(uw) # 1 x 1
            u_hat = u + (muw - uw) * tf.transpose(w) / tf.reduce_sum(w**2)  # 1 x d
            zwb = tf.matmul(z,w) + b  # n x 1

            f_z = z + u_hat * tf.tanh(zwb) #n x d #tf.expand_dims(tf.tanh(zwb), 1) # 

            psi = tf.matmul(1.-tf.tanh(zwb)**2, tf.transpose(w)) # n x d 
            psi_u = tf.matmul(psi, u_hat, transpose_b=True) # n x 1
        
            logdet_jacobian = tf.log(tf.abs(1. + psi_u))
            sum_logdet += logdet_jacobian

            z = f_z

        return f_z, tf.squeeze(sum_logdet)


class MultiVariateGaussian():

    def __init__(self, mu, log_var, is_train=True):

        if not is_train:
            mu = tf.stop_gradient(mu)
            log_var = tf.stop_gradient(log_var)

        self._mu = mu
        self._log_var = log_var
        self._var = tf.exp(log_var)

        self.dim = self._mu.get_shape()[0].value
        assert len(list(self._mu.get_shape())) == 1, 'illegal inputs'


    def log_gradient(self, x):   # A, inverse of covariance matrix
        #A = tf.matrix_inverse(self._cov)
        #A = tf.diag(1. / tf.diag_part(self._cov))
        return -(x - self._mu)  / self._var

    def log_prob(self, x, stop_grad=False):
        if stop_grad:
            mu = tf.stop_gradient(self._mu)
            var = tf.stop_gradient(self._var)
        else:
            mu = self._mu
            var = self._var

        assert len(list(x.get_shape())) == 2, 'illegal inputs'
        x = x - mu
        sum_sq_dist_times_inv_cov = tf.reduce_sum(tf.square(x) / var, axis=1)

        lnD = tf.reduce_sum(tf.log(var))
        ln2piD = tf.log(2 * np.pi) * self.dim
        log_coefficients = ln2piD + lnD

        return -0.5 * sum_sq_dist_times_inv_cov - 0.5 * log_coefficients


    def sample(self, n_samples):
        raw = tf.random_normal([int(n_samples), self.dim])
        ret = self._mu + raw * tf.sqrt(self._var)
        ret.set_shape((int(n_samples), self.dim))
        return ret



class GaussianMixture():

    def __init__(self, n_components, mu, log_var, weights=None, is_train=True):
        assert mu.get_shape().ndims == 2 and log_var.get_shape().ndims == 2, 'illegal inputs'
        self.is_train = is_train
        self.n_components = n_components

        self._mu = mu
        self.dim = self._mu.get_shape()[1].value

        self._log_var = log_var
        self._var = tf.exp(self._log_var)

        if self.is_train:
            assert weights is not None, 'illegal inputs'
            self._weights = tf.nn.softmax(weights)
            self._logits = tf.log(self._weights)
            self._cat = lambda x: gumbel_softmax(self._logits, x) # x: n_samples
        else:
            if weights is None:
                weights = tf.ones(shape=(self.n_components,), dtype=tf.float32)
            self._weights = weights / tf.reduce_sum(weights)
            self._cat = tf.distributions.Categorical(probs=self._weights)


    def _sum_log_exp(self, X, weights, mu, log_var):

        diff = tf.expand_dims(X, 0) - tf.expand_dims(mu, 1)  # c x n x d
        diff_times_inv_cov = diff * tf.expand_dims(1./ tf.exp(log_var), 1)  # c x n x d
        sum_sq_dist_times_inv_cov = tf.reduce_sum(diff_times_inv_cov * diff, axis=2)  # c x n 
        ln2piD = tf.log(2 * np.pi) * self.dim

        lnD = tf.reduce_sum(log_var, axis=1)
        log_coefficients = tf.expand_dims(ln2piD + lnD, 1) # c x 1
        log_components = -0.5 * (log_coefficients + sum_sq_dist_times_inv_cov)  # c x n
        log_weighted = log_components + tf.expand_dims(tf.log(weights), 1)  # c x n + c x 1
        log_shift = tf.expand_dims(tf.reduce_max(log_weighted, 0), 0)
        return log_weighted, log_shift


    def log_gradient(self, X):  

        # X: n_samples x d; mu: c x d; cov: c x d x d
        x_shape = X.get_shape()
        assert len(list(x_shape)) == 2, 'illegal inputs'
    
        def posterior(X):
            log_weighted, log_shift = self._sum_log_exp(X, self._weights, self._mu, self._log_var)
            prob = tf.exp(log_weighted - log_shift) # c x n
            prob = prob / tf.reduce_sum(prob, axis=0, keep_dims=True)
            return prob
    
        diff = tf.expand_dims(X, 0) - tf.expand_dims(self._mu, 1)  # c x n x d
        diff_times_inv_cov = -diff * tf.expand_dims(1./self._var, 1)  # c x n x d
    
        P = posterior(X)  # c x n
        score = tf.matmul(
            tf.expand_dims(tf.transpose(P, perm=[1, 0]), 1), # n x 1 x c
            tf.transpose(diff_times_inv_cov, perm=[1, 0, 2]) # n x c x d
        ) 
        return tf.squeeze(score, axis=1)


    def log_prob(self, X, stop_grad=False):  
        # X: n_samples x d; 
        x_shape = X.get_shape()
        assert len(list(x_shape)) == 2, 'illegal inputs'

        if stop_grad:
            weights = tf.stop_gradient(self._weights)
            mu = tf.stop_gradient(self._mu)
            log_var = tf.stop_gradient(self._log_var)
        else:
            weights = self._weights
            mu = self._mu
            log_var = self._log_var

        log_weighted, log_shift = self._sum_log_exp(X, weights, mu, log_var)
        exp_log_shifted = tf.exp(log_weighted - log_shift)
        exp_log_shifted_sum = tf.reduce_sum(exp_log_shifted, axis=0, keep_dims=True)
        logp = tf.log(exp_log_shifted_sum) + log_shift # 1 x n
        return tf.squeeze(logp)


    def sample(self, n_samples):
        n_samples = int(n_samples)

        if self.is_train:
            cat_probs = self._cat(n_samples)  # n x c
            agg_mu = tf.reduce_sum(tf.expand_dims(cat_probs, 2) * self._mu, axis=1) # n x d 
            agg_var = tf.reduce_sum(tf.expand_dims(cat_probs, 2) * self._var, axis=1) # n x d

            raw = tf.random_normal([n_samples, self.dim])
            ret = agg_mu + tf.sqrt(agg_var) * raw # n x d 

        else:
            cat_samples = self._cat.sample(n_samples) # n x 1

            samples_raw_indices = array_ops.reshape(
                      math_ops.range(0, n_samples), cat_samples.get_shape().as_list())

            partitioned_samples_indices = data_flow_ops.dynamic_partition(
                      data=samples_raw_indices,
                      partitions=cat_samples,
                      num_partitions=self.n_components)

            samples_class = [None for _ in range(self.n_components)]
            for c in range(self.n_components):
                n_class = array_ops.size(partitioned_samples_indices[c])
                raw = tf.random_normal([n_class, self.dim])
                samples_class_c = self._mu[c] + raw * tf.sqrt(self._var[c])
                samples_class[c] = samples_class_c

            # Stitch back together the samples across the components.
            ret = data_flow_ops.dynamic_stitch(
                            indices=partitioned_samples_indices, data=samples_class)
            ret.set_shape((int(n_samples), self.dim))
        return ret


    def avg_mean_variance(self):
        mu = tf.reduce_mean( tf.expand_dims(self._weights, 1) * self._mu, axis=0 )
        var = 0
        for i in range(self.n_components):
            var += ( self._weights[i] * ((self._mu[i] - mu)**2 + self._var[i]) )
        #samples = self.sample(5000)
        #_, var= tf.nn.moments(samples, axes=0)
        return self._mu, tf.reduce_mean(var)


def main(_):

    with tf.Graph().as_default(), tf.Session() as sess:

        n_components = 2
        dim = 2

        # gaussian models
        X0 = tf.constant(np.asarray([[1, 2], [1, 3], [1,4]]), dtype=tf.float32)

        #mu0 = tf.constant(np.asarray([1, 2]), dtype=tf.float32)
        #log_var0= tf.constant(np.asarray([1, 2]), dtype=tf.float32)
        #model = MultiVariateGaussian(mu0, log_var0)
        #print( sess.run([model._mu, model._var]) )
        #print( sess.run(model.log_prob(X0)) )
        #print( sess.run(model.log_gradient(X0)) )
        #sample = model.sample(2000)
        #_sample = sess.run(sample)
        #print(np.mean(_sample, axis=0), np.var(_sample, 0))
        #sys.exit(0)

        mu0 = tf.constant(np.asarray([[1, 0], [0, 2]]), dtype=tf.float32)
        log_var0 = tf.constant(np.asarray([[1, 2], [2, 3]]), dtype=tf.float32)
        weights0 = tf.constant(np.asarray([0.2, 0.8]), dtype=tf.float32)

        model = GaussianMixture(2, mu0, log_var0, weights0, is_train=False)
        print(sess.run(model._mu))
        print(sess.run(model._var))
        print(sess.run(model._weights))
        print(sess.run(model.log_prob(X0)))
        print(sess.run(model.log_gradient(X0)))
        sample = model.sample(5000)
        _sample = sess.run(sample)
        print(np.mean(_sample, axis=0), np.mean(np.var(_sample, 0)))
        print(sess.run(model.avg_mean_variance()))

if __name__ == "__main__":
    tf.app.run()
