from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

from util import log, shuffle
import os
import time

import sys
sys.path.append('../')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

'''
    Augmented Lagrangian: for stable training
'''
class Trainer(object):


    def optimize_adagrad(self, train_vars, train_grads, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)  #adagrad with momentum
        train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        #train_op = optimizer.minimize(loss, var_list=train_vars,
        #            global_step=self.global_step,
        #            gate_gradients=optimizer.GATE_NONE)
        return train_op


    def optimize_adam(self, train_vars, train_grads, lr=1e-2):
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(train_grads, train_vars))
        return train_op


    def __init__(self, config, session):
        self.config = config
        self.session = session

        if config.method != 'svgd':
            assert self.config.kernel != 'none', 'illegal inputs'

        self.filepath = '%s_%s_%d_%s_%s_%d' % (
            config.method,
            config.kernel,
            config.n_components,
            repr(config.t),
            repr(config.temperature),
            config.seed,
        )

        #self.train_dir = './train_dir/%s' % self.filepath
        #self.fig_dir = './figures/%s' % self.filepath
        self.res_dir = './results/%s' % self.filepath

        #for folder in [self.train_dir, self.fig_dir]:
        import glob
        for folder in [self.res_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            # clean
            files = glob.glob(folder + '/*')
            for f in files: os.remove(f)

        #log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)
        from model_svgd import Model
        self.model = Model(config)

        # --- optimizer ---
        self.global_step = tf.Variable(0, name="global_step")

        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
        )

        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        #self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.summary_writer = tf.summary.FileWriter(self.res_dir)
        self.checkpoint_secs = 300  # 5 min

        if self.config.method == 'svgd':
            self.train_op = self.optimize_adagrad(self.model.train_vars, 
                train_grads=self.model.train_grads, lr=self.learning_rate)
        else:
            raise NotImplementedError

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def evaluate_step(self, x_train, mus_star, log_vars_star, w_star, sample_size=500, xlim=30):
        assert mus_star.shape == log_vars_star.shape, 'illegal inputs'

        assert self.config.dim == 1, 'toy example, set dim = 1'
        log_pdf = lambda x,m,v: -0.5*np.log(2*np.pi*v) - (0.5 /v)*((x-m)**2)

        xvals = np.linspace(-xlim, xlim, 1000)

        ''' plot the curve of density function '''
        mus, log_vars = self.session.run([self.model.mus, self.model.log_vars])
        assert mus.shape == log_vars.shape, 'illegal inputs'
        samples, probs = [], []
        for m, lv in zip(mus, log_vars):
            s = np.random.normal(m, np.sqrt(np.exp(lv)), sample_size)
            samples.append(s)
            probs.append( np.exp(log_pdf(xvals, m, np.exp(lv))) )

        prob_star = []
        for m, lv in zip(mus_star, log_vars_star):
            prob_star.append( np.exp(log_pdf(xvals, m, np.exp(lv))) )

        ## training log likelihood
        log_prob = self.session.run( self.model.log_prob, feed_dict=self.model.get_feed_dict(x_train) )
        ll = np.mean( np.exp(log_prob) )

        samples = np.concatenate(samples, axis=0)
        ## KL divergence
        kl_qp = 0
        for m, lv, m_star, lv_star in zip(mus, log_vars, mus_star, log_vars_star):
            log_q = log_pdf(samples, m, np.exp(lv))
            log_p = log_pdf(samples, m_star, np.exp(lv_star))
            kl_qp += (log_q - log_p)
        kl_qp = np.mean(kl_qp) / self.config.n_components

        kl_pq = 0
        for m, lv, m_star, lv_star in zip(mus, log_vars, mus_star, log_vars_star):
            log_p = log_pdf(x_train, m_star, np.exp(lv_star))
            log_q = log_pdf(x_train, m, np.exp(lv))
            kl_pq += (log_p - log_q)
        kl_pq = np.mean(kl_pq) / self.config.n_components

        # mode distance
        from scipy.spatial import distance
        dist = distance.cdist(np.reshape(mus_star, (-1, 1)), np.reshape(mus, (-1, 1)), 'euclidean')
        mode_dist = np.mean( np.min(dist, axis=1) )

        pred_mean, pred_var = np.mean(mus), np.mean((mus - np.mean(mus))**2 + np.exp(log_vars))
        true_mean, true_var = np.mean(mus_star), np.mean((mus_star - np.mean(mus_star))**2 + np.exp(log_vars_star))

        mean_diff = (pred_mean - true_mean)**2
        var_ratio = pred_var / true_var
        return samples, xvals, np.stack(probs, axis=0), np.sum(np.expand_dims(w_star, 1) * np.stack(prob_star, axis=0), axis=0), ll, kl_qp, kl_pq, mode_dist, mean_diff, var_ratio


    def train(self):

        log.infov("Training Starts!")
        output_save_step = 1000
        self.session.run(self.global_step.assign(0)) # reset global step

        from load import generate_sample_data
        x_train, mus_star, log_vars_star, w_star = generate_sample_data( k=self.config.n_components, t=self.config.t )
        n_train = len(x_train)

        n_plot = 0
        with open(self.res_dir + "/step.txt", 'w') as f:

            for n_updates in range(1, 1+self.config.max_steps):

                batch_x = x_train[np.random.choice(n_train, self.config.batch_size, replace=False)]
                step, summary, step_time = self.run_single_step(batch_x)

                self.summary_writer.add_summary(summary, global_step=step)


                if n_updates == 1 or n_updates % 500 == 0:
                    samples, xvals, probs, probs_star, ll, kl_qp, kl_pq, mode_dist, mean_diff, var_ratio = \
                                    self.evaluate_step(x_train, mus_star, log_vars_star, w_star, sample_size=500, xlim= self.config.t*self.config.n_components+5)

                    prefix = '%d,%d,%s,%s' % (n_updates, self.config.n_components, self.config.kernel, repr(self.config.temperature))
                    f.write(prefix + ',' + repr(mode_dist) + ',' + repr(mean_diff) + ',' + repr(var_ratio) + ',' + repr(ll) + ',' + repr(kl_qp) + ',' + repr(kl_pq) + '\n')
                    print(prefix + ',' + repr(mode_dist) + ',' + repr(mean_diff) + ',' + repr(var_ratio) + ',' + repr(ll) + ',' + repr(kl_qp) + ',' + repr(kl_pq))

                # plot the density function at the end
                if n_updates == 1 or n_updates == self.config.max_steps:
                    _, ax1 = plt.subplots()
                    plt.plot(xvals, probs_star, '-r', color='r', linewidth=2)
                    plt.plot(xvals, np.mean(probs, axis=0), '-b', linewidth=2)
                    plt.fill_between(xvals, probs_star, alpha=0.5, color='tomato')
                    plt.fill_between(xvals, np.mean(probs, axis=0), alpha=0.5, color='dodgerblue')
                    #for kk in range(self.config.n_components):
                    #    plt.plot(xvals, probs[kk], '--k', linewidth=1)

                    plt.ylim(0.0, 0.14)
                    pp = PdfPages('%s/step_%d.pdf' % (self.res_dir, n_updates))
                    pp.savefig()
                    pp.close()

                    plt.close()
                    n_plot += 1

        ## save model at the end
        #self.saver.save(self.session,
        #        os.path.join(self.res_dir, 'model'),
        #        global_step=step)


    def run_single_step(self, batch_x):
        _start_time = time.time()

        fetch = [self.global_step, self.summary_op, self.train_op, self.model.clip_op]
        fetch_values = self.session.run( fetch, feed_dict=self.model.get_feed_dict(batch_x))
        [step, summary] = fetch_values[:2]

        _end_time = time.time()

        return step, summary, (_end_time - _start_time)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='svgd', required=True, choices=['svgd'])
    parser.add_argument('--max_steps', type=int, default=20000, required=False)
    parser.add_argument('--batch_size', type=int, default=512, required=False)
    parser.add_argument('--dim', type=int, default=1, required=False)
    parser.add_argument('--temperature', type=float, default=1., required=True)
    parser.add_argument('--n_components', type=int, default=4, required=False)
    parser.add_argument('--t', type=float, default=4., required=False)
    parser.add_argument('--kernel', type=str, default='rbf', required=True, choices=['rbf', 'js', 'none'])
    parser.add_argument('--checkpoint', type=str, default=None, required=False)
    parser.add_argument('--learning_rate', type=float, default=5e-4, required=True)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123, required=False)
    config = parser.parse_args()

    if not config.save:
        log.warning("nothing will be saved.")

    np.random.seed(config.seed)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        # intra_op_parallelism_threads=1,
        # inter_op_parallelism_threads=1,
        gpu_options=tf.GPUOptions(allow_growth=True),
        #device_count={'GPU': 1},
    )

    with tf.Graph().as_default(), tf.Session(config=session_config) as sess:
        with tf.device('/cpu:0'):

            tf.set_random_seed(config.seed)
            np.random.seed(config.seed)

            trainer = Trainer(config, sess)
            trainer.train()

if __name__ == '__main__':
    main()

