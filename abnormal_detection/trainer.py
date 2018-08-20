from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pprint import pprint
import sys

import os
import time

from tqdm import tqdm
from util import log, shuffle, iter_data

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

'''
    Augmented Lagrangian: for stable training
'''
class Trainer(object):

    def optimize_adagrad(self, train_grads, train_vars, lr=1e-2):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9)  #adagrad with momentum
        clip_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in zip(train_grads, train_vars)]
        train_op = optimizer.apply_gradients(clip_grads)
        return train_op


    def optimize_adam(self, train_grads, train_vars, lr=1e-2):
        optimizer = tf.train.AdamOptimizer(lr)
        #grads = optimizer.compute_gradients(loss)
        clip_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in zip(train_grads, train_vars)]
        train_op = optimizer.apply_gradients(clip_grads)

        return train_op


    def __init__(self, config, session):
        self.config = config
        self.session = session

        self.filepath = '%s_%d_%s_%s_%d' % (
            config.method,
            config.n_components,
            config.kernel,
            repr(config.temperature),
            config.seed,
        )
        self.res_dir = './results/%s' % self.filepath

        #for folder in [self.train_dir, self.fig_dir]:
        import glob
        for folder in [self.res_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)
            ## clean
            if self.config.clean:
                files = glob.glob(folder + '/*')
                for f in files: os.remove(f)

        #log.infov("Train Dir: %s, Figure Dir: %s", self.train_dir, self.fig_dir)
        if self.config.method == 'svgd':
            from model_svgd import SVGD
            self.model = SVGD(config)
        else:
            raise NotImplementedError

        # --- optimizer ---
        self.global_step = tf.Variable(0, name="global_step")
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=4000,
                decay_rate=0.8,
                staircase=True,
                name='decaying_learning_rate'
        )


        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=1)
        #self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.summary_writer = tf.summary.FileWriter(self.res_dir)
        self.checkpoint_secs = 300  # 5 min

        #self.train_op = self.optimize_adam(self.model.train_grads, self.model.train_vars, lr=self.learning_rate)
        self.train_op = self.optimize_adagrad(self.model.train_grads, self.model.train_vars, lr=self.learning_rate)

        tf.global_variables_initializer().run()
        if config.checkpoint is not None:
            self.ckpt_path = tf.train.latest_checkpoint(self.config.checkpoint)
            if self.ckpt_path is not None:
                log.info("Checkpoint path: %s", self.ckpt_path)
                self.saver.restore(self.session, self.ckpt_path)
                log.info("Loaded the pretrain parameters from the provided checkpoint path")


    def evaluate(self, x_train, y_train, x_test, y_test):
        
        def _compute_energy(X):
            energy = []
            n_x = len(X)
            max_batches = n_x // self.config.batch_size
            if n_x % self.config.batch_size != 0: max_batches+=1 
            for x_batch in tqdm(iter_data(X, size=self.config.batch_size), total=max_batches):
                #z = self.session.run(self.model.z, feed_dict=self.model.get_feed_dict(x_batch))
                #energy.append( self.session.run(self.model.compute_energy(z, phi, mu, scale)) )
                energy.append( self.session.run(self.model.energy, feed_dict=self.model.get_feed_dict(x_batch)) )
            return np.concatenate(energy)

        eng_train = _compute_energy(x_train)
        eng_test = _compute_energy(x_test)
        assert len(eng_train) == len(x_train) and len(eng_test) == len(x_test), 'double check'

        combined_energy = np.concatenate((eng_train, eng_test))
        thresh = np.percentile(combined_energy, 100 - 20)

        pred = (eng_test > thresh).astype(int)
        gt = y_test.astype(int)

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        print("Seed : {:3d}, Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(self.config.seed, accuracy, precision, recall, f_score))
        return accuracy, precision, recall, f_score


    def train(self):
        log.infov("Training Starts!")
        output_save_step = 1000
        self.session.run(self.global_step.assign(0)) # reset global step

        from data_loader import load_kdd99
        x_train, x_test, y_train, y_test = load_kdd99('kdd_cup.npz', self.config.seed)


        n_updates = 0
        with open(self.res_dir + "/step.txt", 'w') as f:
            for e in range(1, 1+self.config.n_epochs):
                x_train, y_train = shuffle(x_train, y_train)
                n_train = len(x_train)
                max_batches = n_train // self.config.batch_size 
                #if n_train % self.config.batch_size != 0: max_batches+=1 

                for x_batch, y_batch in tqdm(iter_data(x_train, y_train, size=self.config.batch_size), total=max_batches):
                    step, summary, loss, step_time = self.run_single_step(x_batch)
                    self.summary_writer.add_summary(summary, global_step=n_updates)

                    n_updates += 1
                    #if n_updates % 100 == 0:
                    #    eng, eng_chk = self.session.run([self.model.energy, self.model.energy_check], feed_dict=self.model.get_feed_dict(x_batch))
                    #    print(np.mean(eng), np.mean(eng_chk))

                if e % 10 == 0:
                    accuracy, precision, recall, f_score = self.evaluate(x_train, y_train, x_test, y_test)
                    f.write(self.filepath + ',' + repr(e) + ',' + repr(accuracy) + ',' + repr(precision) + ',' + repr(recall) + ',' + repr(f_score) + '\n')
                    f.flush()

                    # save model at the end
                    self.saver.save(self.session,
                            os.path.join(self.res_dir, 'model'),
                            global_step=step)


    def run_single_step(self, x_batch):
        _start_time = time.time()

        fetch = [self.global_step, self.summary_op, self.model.loss, self.train_op]
        fetch_values = self.session.run( fetch, feed_dict=self.model.get_feed_dict(x_batch))
        [step, summary, loss] = fetch_values[:3]

        _end_time = time.time()

        return step, summary, loss, (_end_time - _start_time)


    def log_step_message(self, step, loss, step_time, is_train=True):
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "loss: {loss:.4f} " +
                "({sec_per_batch:.3f} sec/batch)"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step, loss=loss,
                         sec_per_batch=step_time,
                         )
               )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='svgd', required=True)
    parser.add_argument('--n_epochs', type=int, default=200, required=False)
    parser.add_argument('--batch_size', type=int, default=1024, required=False)
    parser.add_argument('--dim', type=int, default=118, required=False)
    parser.add_argument('--z_dim', type=int, default=3, required=False)
    parser.add_argument('--n_components', type=int, default=4, required=False)
    parser.add_argument('--lambda_energy', type=float, default=0.1, required=False)
    parser.add_argument('--lambda_cov_inv', type=float, default=0.005, required=False)
    parser.add_argument('--temperature', type=float, default=0.5, required=False)
    parser.add_argument('--kernel', type=str, default='none', required=False, choices=['rbf', 'none'])
    parser.add_argument('--checkpoint', type=str, default='./model_inits', required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--clean', action='store_true', default=False)
    parser.add_argument('--lr_weight_decay', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=1)
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
            assert tf.__version__.startswith('1.8'), 'use tf 1.8 to reproduce the resutls'
            #print('use tf1.8 to reproduce the results')
            #tf.set_random_seed(100)

            np.random.seed(config.seed)
            trainer = Trainer(config, sess)
            trainer.train()

if __name__ == '__main__':
    main()

