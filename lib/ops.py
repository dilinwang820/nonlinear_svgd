from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import sys
import numpy as np

from tqdm import tqdm

def sqr_dist(x, y, e=1e-8):

    assert len(list(x.get_shape())) == 2 and len(list(y.get_shape()))==2, 'illegal inputs'
    xx = tf.reduce_sum(tf.square(x) + 1e-10, axis=1)
    yy = tf.reduce_sum(tf.square(y) + 1e-10, axis=1)
    xy = tf.matmul(x, y, transpose_b=True)
    dist = tf.expand_dims(xx, 1) + tf.expand_dims(yy, 0) - 2.* xy
    return dist


def median_distance(H):
    V = tf.reshape(H, [-1])
    n = tf.size(V)
    top_k, _ = tf.nn.top_k(V, k= (n // 2) + 1)
    return tf.cond(
        tf.equal(n%2, 0),
        lambda: (top_k[-1] + top_k[-2]) / 2.0,
        lambda: top_k[-1]
    )
    return h


def poly_kernel(x, subtract_mean=True, e=1e-8):
    if subtract_mean:
        x = x - tf.reduce_mean(x, axis=0)
    kxy = 1 + tf.matmul(x, x, transpose_b=True)
    kxkxy = x * x.get_shape()[0]
    return kxy, dxkxy


def rbf_kernel(x, h=-1):

    H = sqr_dist(x, x)
    if h == -1:
        h = tf.maximum(1e-6, median_distance(H))

    kxy = tf.exp(-H / h)
    dxkxy = -tf.matmul(kxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy


def imq_kernel(x, h=-1):
    H = sqr_dist(x, x)
    if h == -1:
        h = median_distance(H)

    kxy = 1. / tf.sqrt(1. + H / h) 

    dxkxy = .5 * kxy / (1. + H / h)
    dxkxy = -tf.matmul(dxkxy, x)
    sumkxy = tf.reduce_sum(kxy, axis=1, keep_dims=True)
    dxkxy = (dxkxy + x * sumkxy) * 2. / h

    return kxy, dxkxy



def kernelized_stein_discrepancy(X, score_q, kernel='rbf', h=-1, **model_params):
    n, dim = tf.cast(tf.shape(X)[0], tf.float32), tf.cast(tf.shape(X)[1], tf.float32)
    Sqx = score_q(X, **model_params)

    H = sqr_dist(X, X)
    if h == -1:
        h = median_distance(H) # 2sigma^2
    h = tf.sqrt(h/2.)
    # compute the rbf kernel
    Kxy = tf.exp(-H / h ** 2 / 2.)

    Sqxdy = -(tf.matmul(Sqx, X, transpose_b=True) - tf.reduce_sum(Sqx * X, axis=1, keep_dims=True)) / (h ** 2)

    dxSqy = tf.transpose(Sqxdy)
    dxdy = (-H / (h ** 4) + dim / (h ** 2))

    M = (tf.matmul(Sqx, Sqx, transpose_b=True) + Sqxdy + dxSqy + dxdy) * Kxy 
    #M2 = M - T.diag(T.diag(M)) 

    #ksd_u = tf.reduce_sum(M2) / (n * (n - 1)) 
    #ksd_v = tf.reduce_sum(M) / (n ** 2) 

    #return ksd_v
    return M


def svgd_gradient(x, grad, kernel='rbf', temperature=1., u_kernel=None, **kernel_params):
    assert x.get_shape() == grad.get_shape(), 'illegal inputs and grads'
    p_shape = tf.shape(x)
    if tf.keras.backend.ndim(x) > 2:
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        grad = tf.reshape(grad, (tf.shape(grad)[0], -1))

    if kernel == 'rbf':
        kxy, dxkxy = rbf_kernel(x, **kernel_params)
    elif kernel == 'poly':
        kxy, dxkxy = poly_kernel(x)
    elif kernel == 'js':
        kxy, dxkxy = u_kernel['kxy'], u_kernel['dxkxy']
    elif kernel == 'imq':
        kxy, dxkxy = imq_kernel(x)
    elif kernel == 'none':
        kxy = tf.eye(tf.shape(x)[0])
        dxkxy = tf.zeros_like(x)
    else:
        raise NotImplementedError

    svgd_grad = (tf.matmul(kxy, grad) + temperature * dxkxy) / tf.reduce_sum(kxy, axis=1, keep_dims=True)

    svgd_grad = tf.reshape(svgd_grad, p_shape)
    return svgd_grad


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(inputs, num_outputs, activation_fn=tf.nn.relu,
           kernel_size=5, stride=2, padding='SAME', name="conv2d"):

    with tf.variable_scope(name):
        return tf.contrib.layers.conv2d( inputs, num_outputs, kernel_size, stride=stride, padding=padding, activation_fn=activation_fn)


def deconv2d(inputs, num_outputs, activation_fn=tf.nn.relu,
        kernel_size=5, stride=2, padding='SAME', name="deconv2d"):

    with tf.variable_scope(name):
        return tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride=stride, padding=padding, activation_fn=activation_fn)


def fc(input, output_shape, activation_fn=tf.nn.relu, init=None, name="fc"):
    if init is None: init = tf.glorot_uniform_initializer()
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn, weights_initializer=init)
    return output


