from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint

import os
import time

from tqdm import tqdm
from util import log
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from clustering_ops import *
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from lib.vis import grayscale_grid_vis

'''
    Augmented Lagrangian: for stable training
'''
class Trainer(object):

    def optimize_adagrad(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)  #adagrad with momentum
        if train_grads is not None:
            clip_grads = [(tf.clip_by_norm(grad, 20), var) for grad, var in zip(train_grads, train_vars)]
            #train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
            train_op = optimizer.apply_gradients(clip_grads)
        else:
            train_op = optimizer.minimize(tf.reduce_mean(loss), var_list=train_vars, global_step=self.global_step)
        return train_op


    def optimize_adam(self, train_vars, loss=None, train_grads=None, lr=1e-2):
        assert (loss is not None) or (train_grads is not None), 'illegal inputs'
        optimizer = tf.train.AdamOptimizer(lr)
        if train_grads is not None:
            clip_grads = [(tf.clip_by_norm(grad, 20), var) for grad, var in zip(train_grads, train_vars)]
            #train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
            train_op = optimizer.apply_gradients(clip_grads)
        else:
            train_op = optimizer.minimize(loss, var_list=train_vars, global_step=self.global_step)
        return train_op


    def __init__(self, config, session):
        self.config = config
        self.session = session

        if self.config.method == 'svgd':
            self.filepath = '%s_%s_%s_%s_%d' % (
                config.method,
                config.dataset,
                config.kernel,
                repr(config.temperature),
                config.seed
            )
        else:
            self.filepath = '%s_%s' % (config.method, config.dataset)

        self.res_dir = './results/%s/' % self.filepath
        self.fig_dir = './results/%s/figures'  % self.filepath
        self.res_gmm_dir = './results/%s/gmm'  % self.filepath
        self.res_pretrain_dir = './results/%s_%s_pretrain'  % (self.config.method, self.config.dataset)

        #for folder in [self.train_dir, self.fig_dir]:
        import glob
        for folder in [self.res_dir, self.fig_dir, self.res_gmm_dir, self.res_pretrain_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            ### clean
            ###if self.config.clean:
            #files = glob.glob(folder + '/events.*') + glob.glob(folder + '/*.png')
            #for f in files: os.remove(f)
                
        #log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)
        if self.config.method == 'svgd':
            from model_svgd import SVGD
            self.model = SVGD(config)
        else:
            raise NotImplementedError

        # --- optimizer ---
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.learning_rate = config.learning_rate

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        #self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.summary_writer = tf.summary.FileWriter(self.res_dir)
        self.checkpoint_secs = 300  # 5 min

        ## prtraining op
        self.pre_train_op = self.optimize_adam(self.model.net_train_vars, \
                            loss=self.model.loss_recons_noisy, lr=self.config.learning_rate)

        if self.config.method == 'svgd':
            self.depict_op = self.optimize_adam(self.model.train_vars, \
                            loss=self.model.depict_loss, lr=self.learning_rate)
            self.svgd_op = self.optimize_adam(self.model.gmm_train_vars, \
                            train_grads=self.model.gmm_train_grads, lr=self.config.learning_rate)

        tf.global_variables_initializer().run()


    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        max_batches = len(inputs) // batchsize
        if len(inputs) % batchsize != 0: max_batches += 1
        for i in range(max_batches):
            start_idx = i * batchsize
            end_idx = min(len(inputs), (i+1) * batchsize)
            if shuffle:
                excerpt = indices[start_idx:end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield inputs[excerpt], targets[excerpt], excerpt


    def try_load_checkpoint(self, model_path):

        ckpt_path = tf.train.latest_checkpoint(model_path)
        assert ckpt_path is not None, '%s is empty' % model_path
        log.info("Checkpoint path: %s", ckpt_path)
        self.saver.restore(self.session, ckpt_path)
        log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def save_curr_model(self, model_path):

        step = self.session.run(self.global_step)
        self.saver.save(self.session,
                model_path,
                global_step=step)


    def get_latent_rep_and_pred(self, inputs, targets, batch_size=100):
        y_pred, latent_z = [], []
        for batch in self.iterate_minibatches(inputs, targets, batch_size, shuffle=False):
            x_batch, _, _ = batch
            pred, z = self.session.run( [self.model.pred_clean, self.model.z], feed_dict=self.model.get_feed_dict(x_batch))
            y_pred.append(pred)
            latent_z.append(z)
        y_pred = np.concatenate(y_pred, axis=0)
        latent_z = np.concatenate(latent_z, axis=0)

        return latent_z, np.argmax(y_pred, 1)


    ''' pre-training auto encoder ''' 
    def pre_train_enc_dec(self, inputs, targets, num_epochs=1000, batch_size=100):
        for epoch in range(1, num_epochs + 1):
            train_err = 0
            for batch in self.iterate_minibatches(inputs, targets, batch_size, shuffle=True):
                x_batch, _, _ = batch
                err, _ = self.session.run( [self.model.loss_recons_clean, self.pre_train_op], \
                            feed_dict=self.model.get_feed_dict(x_batch))
                train_err += err
            log.info(("pre-training autoencoder epoch: {:d}, loss:{:4f}").format(epoch, train_err))

            if epoch % (num_epochs // 10) == 0:
                self.save_curr_model(os.path.join(self.res_pretrain_dir, 'model'))  #save model
                latent_z, _ = self.get_latent_rep_and_pred(inputs, targets)
                y_pred, _ = clustering(latent_z, self.config.num_clusters)
                metrics(targets, y_pred)


    def train_svgd(self, inputs, targets, batch_size=100, num_epochs=4000): 

        def normalize(y_prob):
            cluster_frequency = np.sum(y_prob, axis=0)
            y_prob = y_prob ** 2 / cluster_frequency
            y_prob = np.transpose(y_prob.T / np.sum(y_prob, axis=1))
            y_pred = np.argmax(y_prob, axis=1)
            return y_prob, y_pred

        n_train = len(inputs)
        y_prob = np.zeros((n_train, self.config.num_clusters))
        y_prob_prev = np.zeros((n_train, self.config.num_clusters))

        for batch in self.iterate_minibatches(inputs, targets, batch_size, shuffle=False):
            x_batch, _, idx_batch = batch #TODO
            minibatch_prob = self.session.run(self.model.pred_clean, feed_dict=self.model.get_feed_dict(x_batch))
            y_prob[idx_batch] = minibatch_prob

        if True:
            y_prob, y_pred = normalize(y_prob)

        n_updates = 0
        for epoch in range(1, num_epochs+1):

            recon_loss_iter, clus_loss_iter, loss_iter, energy_iter = 0., 0., 0, 0
            for batch in self.iterate_minibatches(inputs, targets, batch_size, shuffle=True):
                x_batch, _, idx_batch = batch

                fetch_values = [self.model.loss, self.model.loss_recons_noisy, self.model.loss_clus, self.model.energy_noisy, self.summary_op, self.depict_op]
                if epoch > 20: fetch_values.append(self.svgd_op)

                ret = \
                            self.session.run( fetch_values, feed_dict=self.model.get_feed_dict(x_batch, y_prob[idx_batch]))

                loss, loss_recons, loss_clus, energy, summary = ret[:5]

                minibatch_prob = self.session.run(self.model.pred_clean, feed_dict=self.model.get_feed_dict(x_batch))
                y_prob[idx_batch] = minibatch_prob

                loss_iter += loss
                recon_loss_iter += loss_recons
                clus_loss_iter += loss_clus
                energy_iter += energy

                self.summary_writer.add_summary(summary, global_step=n_updates)
                n_updates += 1

            print(epoch, 'recon_loss', recon_loss_iter, 'clus_loss', clus_loss_iter, 'loss', loss_iter, 'energy', energy_iter)
            print(epoch, metrics(targets, y_pred) )

            if True:
                y_prob, y_pred = normalize(y_prob)
            if np.sum( (y_pred - np.argmax(y_prob_prev, axis=1)) ** 2) < 1e-6:
                break
            y_prob_prev = np.copy(y_prob)

            if epoch % 10 == 0:
                latent_z, y_pred = self.get_latent_rep_and_pred(inputs, targets)
                plot_latent_z_space(latent_z, y_pred, '%s/step-%d.png' % (self.res_dir, epoch))

        print(epoch, metrics(targets, y_pred) )


    def train(self):

        log.infov("Training Starts!")
        output_save_step = 1000
        self.session.run(self.global_step.assign(0)) # reset global step

        if self.config.dataset == 'mnist':
            from load import load_mnist
            inputs, targets = load_mnist()
        else:
            raise NotImplementedError

        if self.config.method == 'kmeans': 
            y_pred, _ = clustering(np.reshape(inputs, (len(inputs), -1)), self.config.num_clusters)
            metrics(targets, y_pred)
            return

        ''' pre-training '''
        if not self.config.skip_pretrain:
            self.pre_train_enc_dec( inputs, targets, batch_size=self.config.batch_size, num_epochs=1000)
            # save model
            self.save_curr_model(os.path.join(self.res_pretrain_dir, 'model'))
        else:
            self.try_load_checkpoint(self.res_pretrain_dir)

        # plot
        latent_z, _ = self.get_latent_rep_and_pred(inputs, targets)
        y_pred, centroids = clustering(latent_z, self.config.num_clusters)
        plot_latent_z_space(latent_z, y_pred, \
                    '%s/pre_train_z' % self.res_dir, with_legend=True)
        #sys.exit(0) 

        if self.config.method == 'svgd':
            if not self.config.skip_svgd:
                self.session.run( self.model.mu.assign(centroids) )
                #scale = np.zeros((self.config.num_clusters,  self.config.z_dim*(self.config.z_dim+1)//2))
                scale = np.zeros((self.config.num_clusters,  self.config.z_dim))
                for c in range(self.config.num_clusters):
                    z_c = latent_z[np.where(y_pred == c)[0]]
                    s0 = np.std(z_c, axis=0)
                    scale[c] = s0
                self.session.run( self.model.scale_diag.assign(scale))

                self.train_svgd(inputs, targets, num_epochs=400, batch_size=self.config.batch_size)
                self.save_curr_model(os.path.join(self.res_dir, 'model'))
            else:
                self.try_load_checkpoint(self.res_dir)

        # plot
        latent_z, y_pred = self.get_latent_rep_and_pred(inputs, targets)
        #y_pred, centroids = clustering(latent_z, self.config.num_clusters)
        plot_latent_z_space(latent_z, y_pred, \
                    '%s/%s_z' % (self.res_dir, self.config.method), with_legend=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='svgd', required=False, choices=['svgd'])
    parser.add_argument('--dataset', type=str, default='mnist', required=False, choices=['mnist'])
    parser.add_argument('--n_epochs', type=int, default=1000, required=False)
    parser.add_argument('--batch_size', type=int, default=100, required=False)
    parser.add_argument('--dim', type=int, default=28, required=False)
    parser.add_argument('--c', type=int, default=1, required=False)
    parser.add_argument('--z_dim', type=int, default=10, required=False)
    parser.add_argument('--num_clusters', type=int, default=10, required=False)
    parser.add_argument('--n_components', type=int, default=10, required=False)
    parser.add_argument('--temperature', type=float, default=0.5, required=False)
    parser.add_argument('--kernel', type=str, default='none', required=False, choices=['rbf', 'none'])
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False)
    parser.add_argument('--hlayer_loss_param', type=float, default=0.01, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--skip_pretrain', action='store_true', default=False)
    parser.add_argument('--skip_svgd', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123, required=False)
    config = parser.parse_args()

    if not config.save:
        log.warning("nothing will be saved.")

    if config.kernel != 'none':
        assert config.temperature > 0., 'illegal temperature'

    np.random.seed(config.seed)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        # intra_op_parallelism_threads=1,
        # inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(allow_growth=True),
        #device_count={'GPU': 1},
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/device:GPU:%d'% config.gpu):
            tf.set_random_seed(config.seed)
            np.random.seed(config.seed)

            trainer = Trainer(config, sess)
            trainer.train()


if __name__ == '__main__':
    main()

