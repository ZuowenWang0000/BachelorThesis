"""
Evaluation of a given checkpoint in the standard and adversarial sense.  Can be
called as an infinite loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
from datetime import datetime
from itertools import product, repeat
import json
import math
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import trange

sys.path.append("..")
import cifar10_input
import cifar100_input
import svhn_input
import resnet
import vgg
from spatial_attack import SpatialAttack
import utilities


def exp_over_grid_eval_random_sample(model, sess, config, eval_on_train=False, 
                                     num_eval_examples=100, seed=1):
    np.random.seed(seed)
    if config.data.dataset_name == "cifar-10":
        data_iterator = cifar10_input.CIFAR10Data(config.data.data_path)
    elif config.data.dataset_name == "cifar-100":
        data_iterator = cifar100_input.CIFAR100Data(config.data.data_path)
    elif config.data.dataset_name == "svhn":
        data_iterator = svhn_input.SVHNData(config.data.data_path)
    else:
        raise ValueError("Unknown dataset name.")
    
    if eval_on_train:
        n = data_iterator.train_data.n
        indices = np.random.choice(n, num_eval_examples)
        x_batch = data_iterator.train_data.xs[indices, :]
        y_batch = data_iterator.train_data.ys[indices]
    else:
        n = data_iterator.eval_data.n
        indices = np.random.choice(n, num_eval_examples)
        x_batch = data_iterator.eval_data.xs[indices, :]
        y_batch = data_iterator.eval_data.ys[indices]

    grid = product(*list(np.linspace(-l, l, num=g) 
        for l, g in zip(config.attack.spatial_limits, 
        config.attack.grid_granularity)))
    n_grid = np.prod(config.attack.grid_granularity)
    results = np.zeros([len(x_batch), n_grid])
    n_batch = len(x_batch)
    for i, (tx, ty, r) in enumerate(grid):
        trans = np.stack(repeat([tx, ty, r], n_batch))
        dict_nat = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.transform: trans,
                    model.is_training: False}
        results[:,i] = sess.run(model.correct_prediction, feed_dict = dict_nat)

    return results


# A function for evaluating a single checkpoint
def exp_over_grid_eval(model, attack, sess, config, attack_type, 
                       eval_on_train=False):
    num_eval_examples = config.eval.num_eval_examples
    eval_batch_size = config.eval.batch_size

    if config.data.dataset_name == "cifar-10":
        data_iterator = cifar10_input.CIFAR10Data(config.data.data_path)
    elif config.data.dataset_name == "cifar-100":
        data_iterator = cifar100_input.CIFAR100Data(config.data.data_path)
    elif config.data.dataset_name == "svhn":
        data_iterator = svhn_input.SVHNData(config.data.data_path)
    else:
        raise ValueError("Unknown dataset name.")

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    n_grid = np.prod(config.attack.grid_granularity)
    results = np.zeros([num_eval_examples, n_grid])
    for ibatch in trange(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        if eval_on_train:
            x_batch = data_iterator.train_data.xs[bstart:bend, :]
            y_batch = data_iterator.train_data.ys[bstart:bend]
        else:
            x_batch = data_iterator.eval_data.xs[bstart:bend, :]
            y_batch = data_iterator.eval_data.ys[bstart:bend]
        grid = product(*list(np.linspace(-l, l, num=g) 
            for l, g in zip(config.attack.spatial_limits, 
            config.attack.grid_granularity)))
        n_batch = len(x_batch)
        for i, (tx, ty, r) in enumerate(grid):
            trans = np.stack(repeat([tx, ty, r], n_batch))
            dict_nat = {model.x_input: x_batch,
                        model.y_input: y_batch,
                        model.transform: trans,
                        model.is_training: False}
            results[bstart:bend,i] = sess.run(
                model.correct_prediction, feed_dict = dict_nat)
   
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Eval script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default="../configs/christinaconfig_cifar10_spatial.json", required=False)
    parser.add_argument('--save_root_path', type=str,
                        help='path to repo dir',
                        default='/Users/heinzec/projects/core-da/repo_dir_7jan', required=False)
    parser.add_argument('--exp_id', type=str,
                        help='ID of experiment to load',
                        default='3e3p7xPG98_1058376', required=False)   
    parser.add_argument('-s', '--save_folder', type=str,
                        help='path to plots folder',
                        default='../draft/plots', required=False)
    parser.add_argument('--save_folder_pkl', type=str,
                        help='path to pkl folder',
                        default='../draft/plots', required=False)                
    parser.add_argument('--method', type=str,
                        help='method name',
                        default='core', required=False)
    parser.add_argument('--seed', type=int,
                        help='random seed',
                        default=1, required=False)
    parser.add_argument('--k', type=int,
                        help='chosen k',
                        default=1, required=False)
    parser.add_argument('--eval_on_train', type=int,
                        help='flag whether to use training or test images',
                        default=0, required=False)
    args = parser.parse_args()
    
    config_dict = utilities.get_config(args.config)
    config_dict_copy = copy.deepcopy(config_dict)
    out_dict = {}
    out_dict['hyperparameters'] = config_dict_copy  

    config = utilities.config_to_namedtuple(config_dict)

    if config.attack.use_spatial:
        attack = config.attack.spatial_method
    elif config.attack.use_linf:
        attack = 'linf'
    dataset = config.data.dataset_name.upper()
    split = 'train' if args.eval_on_train else 'test'

    # create plot subdirectories
    save_plot_subdir = os.path.join(args.save_folder, dataset, attack)
    os.makedirs(save_plot_subdir, exist_ok=True)
    save_folder_pkl =  os.path.join(args.save_folder_pkl, dataset, attack, split)
    os.makedirs(save_folder_pkl, exist_ok=True)

    # num_ids does not matter for eval
    num_ids = 64
    model_family = config.model.model_family
    if model_family == "resnet":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False 
        model = resnet.Model(config.model, num_ids, diffable)
    elif model_family == "vgg":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            # TODO: add differentiable transformer to vgg.py
            raise NotImplementedError
        model = vgg.Model(config.model, num_ids)

    attack_eval_grid = SpatialAttack(model, config.attack, 'grid')                                   

    model_dir = '%s/logdir/%s' % (args.save_root_path, args.exp_id)    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    
    if ckpt is None:
        print('No checkpoint found.')
    else:
        with tf.Session() as sess:
            # Restore the checkpoint
            saver.restore(sess, 
                os.path.join(model_dir, 
                             ckpt.model_checkpoint_path.split("/")[-1]))
            results_matrix = exp_over_grid_eval(model, attack, sess, config, 
                'grid', eval_on_train=args.eval_on_train)
            misclassified = np.zeros([results_matrix.shape[0], results_matrix.shape[1]])
            misclassified[np.where(results_matrix == 0)] = 1
            row_means = np.mean(misclassified, axis=1)
            row_means_cor_class = np.mean(results_matrix, axis=1)

            # plt.figure(2)
            # plt.violinplot(row_means)
            # save_path_plot = os.path.join(save_plot_subdir, 
            #     "{}_{}_{}_misclassified_violin_{}_k{}.pdf".format(
            #         dataset, split, args.exp_id, args.method, args.k))
            # plt.savefig(save_path_plot)
            out_dict['mean'] = np.mean(row_means)
            out_dict['std'] = np.std(row_means)
            save_path_json = os.path.join(save_plot_subdir, 
                "{}_{}_{}_misclassified_stats_{}_k{}.json".format(
                    dataset, split, args.exp_id, args.method, args.k))

            with open(save_path_json, 'w') as result_file:
                 json.dump(out_dict, result_file, sort_keys=True, indent=4)

            save_path_pkl = os.path.join(save_folder_pkl, 
                "{}_{}_{}_misclassified_mat_{}_k{}.pkl".format(
                    dataset, split, args.exp_id, args.method, args.k))
            bytes_to_store = pickle.dumps(misclassified)
            file = open(save_path_pkl, 'wb')
            file.write(bytes_to_store)
            file.close()
