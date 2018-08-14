import sys
sys.path.append('..')

import numpy as np
import os
from time import time
from collections import Counter
import random
from matplotlib import pyplot as plt
import h5py
import scipy.io as sio

from util import shuffle
from lib.vis import grayscale_grid_vis, color_grid_vis

def load_mnist( data_dir = '/home/dilin/data/mnist_data'):
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = (np.concatenate( (trX, teX), axis=0) - 127.5) / 127.5
    y = np.concatenate( (trY, teY), axis=0)
    return X.astype('float32'), y


