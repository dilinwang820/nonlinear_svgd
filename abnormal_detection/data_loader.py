import os
import random
import numpy as np
import collections
import numbers
import math
import pandas as pd

def load_kdd99(data_path, seed):

        np.random.seed(seed)

        data = np.load(data_path)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        N, D = features.shape

        normal_data = features[labels==1]
        normal_labels = labels[labels==1]

        N_normal = normal_data.shape[0]

        attack_data = features[labels==0]
        attack_labels = labels[labels==0]

        N_attack = attack_data.shape[0]

        randIdx = np.arange(N_attack)
        np.random.shuffle(randIdx)
        N_train = N_attack // 2

        train = attack_data[randIdx[:N_train]]
        train_labels = attack_labels[randIdx[:N_train]]

        test = attack_data[randIdx[N_train:]]
        test_labels = attack_labels[randIdx[N_train:]]

        test = np.concatenate((test, normal_data),axis=0)
        test_labels = np.concatenate((test_labels, normal_labels),axis=0)

        # min-max normalization
        norm_idx = data['norm_idx'].astype('int32')

        min_val = np.min(train[:, norm_idx], axis=0)
        max_val = np.max(train[:, norm_idx], axis=0)
        train[:, norm_idx] = (train[:, norm_idx] - min_val) / (max_val - min_val + 1e-8)
        test[:, norm_idx] = (test[:, norm_idx] - min_val) / (max_val - min_val + 1e-8)

        return train, test, train_labels, test_labels


