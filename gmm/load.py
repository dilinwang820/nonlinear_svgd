from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_data(k = 5, t = 4):

    n_c_d = 2000
    x = []

    sigma = 1.
    xl = t * k
    mus_star, log_vars_star = [], []
    for i in np.linspace(-xl, xl, k):
        x.append( np.random.normal(i, sigma, size=(n_c_d)))
        mus_star.append(i)
        log_vars_star.append( np.log(sigma*2) )

    wx = np.concatenate(x, axis=0).astype('float32')
    wx = np.expand_dims(wx, 1)
    np.random.shuffle(wx)
    x_train = wx

    w_star = [1. / k] * k
    return x_train, np.asarray(mus_star), np.asarray(log_vars_star), w_star

