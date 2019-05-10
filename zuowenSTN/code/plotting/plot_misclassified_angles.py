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
from datetime import datetime
import json
import math
import os
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


def plot_images(img, y, pred, imgp, predp, bins, label_names, save_plot_subdir, name, method, 
                id, k):
    save_plot_subdir_subdir = os.path.join(
        save_plot_subdir, "{}_{}_{}_k{}_{}".format(split, name, method, k, id))
    os.makedirs(save_plot_subdir_subdir, exist_ok=True)
    for i in range(img.shape[0]):
        # plot image
        plt.figure(2, figsize=(2,2))
        plt.rc('figure', titlesize=16)  # fontsize of the figure title
        plt.rc('xtick', labelsize=14)
        plt.imshow(img[i,...].astype(np.uint8))
        ax = plt.gca()
        #for ax in axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.title("Truth: {}".format(y[i]))
        plt.savefig(os.path.join(
            save_plot_subdir_subdir, 
            "{}_{}_{}_{}_k{}_{}.pdf".format(dataset, split, name, method, k, i)), 
            pad_inches=0.1, bbox_inches='tight')
        plt.close()

        # plot image
        plt.figure(3, figsize=(2,2))
        plt.imshow(imgp[i,...].astype(np.uint8))
        plt.axis('off')
        ax = plt.gca()
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.title("Pred.: {}".format(predp[i]))
        plt.savefig(os.path.join(
            save_plot_subdir_subdir, 
            "{}_{}_{}_{}_k{}_pert_{}.pdf".format(dataset, split, name, method, k, i)), 
           pad_inches=0.1, bbox_inches='tight')
        plt.close()

        # plot histogram
        plt.figure(4, figsize=(2,2))
        ind_unique, counts = np.unique(pred[i, ...], return_counts=True)
        ind = np.arange(bins)
        counts_bar = [0]*bins
        for j in range(bins):
            idx = np.where(j == ind_unique)[0]
            if len(idx) > 0:
                counts_bar[j] = np.asscalar(counts[idx])
        plt.bar(ind, counts_bar, color = '#1f77b4')
        plt.xticks(ind, label_names, rotation='vertical')
        plt.yticks(np.arange(0, 61, 10))
        plt.title("Truth: {}".format(y[i]))
        plt.savefig(os.path.join(
            save_plot_subdir_subdir, 
            "{}_{}_{}_hist_{}_k{}_{}.pdf".format(
                dataset, split, name, args.method, args.k, i)),
            pad_inches=0.1, bbox_inches='tight')
        plt.close()



def get_images(indices_some_correct, config, data_iterator, model, sess, 
               x_batch, y_batch, results, predictions, angles):
    
    predictions_some_correct = predictions[indices_some_correct,:]
    y_some_correct_num = y_batch[indices_some_correct]

    dict_nat = {model.x_input: x_batch[indices_some_correct, ...],
                model.y_input: y_some_correct_num,
                model.transform: np.zeros([len(indices_some_correct), 3]),
                model.is_training: False}
    img_some_correct = sess.run(model.x_image, feed_dict = dict_nat)
    if (config.data.dataset_name == "cifar-10" or 
        config.data.dataset_name == "cifar-100"):
        y_some_correct_labels = [data_iterator.label_names[y] 
            for y in y_some_correct_num]
    else:
        y_some_correct_labels = y_some_correct_num 

    trans_perturb = np.zeros([len(indices_some_correct), 3])
    pred_perturb = []
    for i, idx in enumerate(indices_some_correct):
        idx_incorrect_angles = np.where(results[idx,:] == 0)[0]
        if len(idx_incorrect_angles) == 0:
            idx_incorrect_angles = np.where(results[idx,:] == 1)[0]
        incorrect_angles = angles[idx_incorrect_angles]
        # pick one of the incorrect angles at random
        idx_chosen_angle = np.random.choice(len(incorrect_angles), size=1)
        chosen_angle = incorrect_angles[idx_chosen_angle]
        trans_perturb[i,2] = np.asscalar(chosen_angle)
        pred_perturb.append(np.asscalar(predictions_some_correct[i,chosen_angle+30]))

    dict_per = {model.x_input: x_batch[indices_some_correct, ...],
                model.y_input: y_some_correct_num,
                model.transform: trans_perturb,
                model.is_training: False}
    img_some_correct_perturb = sess.run(model.x_image, feed_dict = dict_per)
    
    if (config.data.dataset_name == "cifar-10" or 
        config.data.dataset_name == "cifar-100"):
        pred_perturb_labels = [data_iterator.label_names[int(y)] for y in pred_perturb]
    else:
        pred_perturb_labels = pred_perturb 
    
    some_correct_tuple = (img_some_correct, y_some_correct_labels, 
        predictions_some_correct, img_some_correct_perturb, pred_perturb_labels)

    return some_correct_tuple

def get_correctly_classified_angles(model, sess, config, eval_on_train=False, 
                                    num_eval_examples=200, seed=1):
    np.random.seed(seed)
    if config.data.dataset_name == "cifar-10":
        data_iterator = cifar10_input.CIFAR10Data(config.data.data_path)
    elif config.data.dataset_name == "cifar-100":
        data_iterator = cifar100_input.CIFAR100Data(config.data.data_path)
    elif config.data.dataset_name == "svhn":
        data_iterator = svhn_input.SVHNData(config.data.data_path)
    
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

    trans = np.zeros([len(x_batch), 3])
    results = np.zeros([len(x_batch), 61])
    predictions = np.zeros([len(x_batch), 61])
    angles = np.arange(-30, 31)
    for i, j in enumerate(range(-30, 31)):
        trans[:,2] = j
        dict_nat = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.transform: trans,
                    model.is_training: False}
        results[:,i], predictions[:,i] = sess.run(
            [model.correct_prediction, model.predictions], feed_dict = dict_nat)

    cor_class_means = np.mean(results, axis=1)
    # get images that fooled the model for all angles
    indices_all_fooled = np.where(cor_class_means == 0)[0]  
    if len(indices_all_fooled) > 0:
        fooled_tuple = get_images(indices_all_fooled, config, data_iterator, 
            model, sess, x_batch, y_batch, results, predictions, angles)
    else:
        fooled_tuple = None
  
    # get images that were correctly classified for all angles
    indices_all_correct = np.where(cor_class_means == 1)[0]
    if len(indices_all_correct) > 0:
        correct_tuple = get_images(indices_all_correct, config, data_iterator, 
            model, sess, x_batch, y_batch, results, predictions, angles)
    else:
        correct_tuple = None

    # get images that were correctly classified for a large proportion of angles
    indices_some_correct = np.where(
        np.logical_and(cor_class_means >= .9, cor_class_means < 1))[0]
    if len(indices_some_correct) > 0:
        some_correct_tuple = get_images(indices_some_correct, config, data_iterator, 
            model, sess, x_batch, y_batch, results, predictions, angles)
    else:
        some_correct_tuple = None

    return results, predictions, fooled_tuple, correct_tuple, some_correct_tuple


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
                        default='../../draft/plots', required=False)
    parser.add_argument('--dataset', type=str,
                        help='dataset name',
                        default='CIFAR-10', required=False)   
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
    attack = 'spatial'
    dataset = args.dataset
    split = 'train' if args.eval_on_train else 'test'
    # create plot subdirectories
    save_plot_subdir = os.path.join(args.save_folder, dataset, attack)
    os.makedirs(save_plot_subdir, exist_ok=True)
    save_plot_subdir_subdir = os.path.join(
        save_plot_subdir, "{}_angles_k{}".format(split, args.k))
    os.makedirs(save_plot_subdir_subdir, exist_ok=True)
    config_dict = utilities.get_config(args.config)
    config = utilities.config_to_namedtuple(config_dict)
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
            out = get_correctly_classified_angles(
                model, sess, config, eval_on_train=args.eval_on_train, 
                seed=args.seed)
            (results_matrix, predictions, fool_tup, cor_tup, some_cor_tup) = out 
            print(results_matrix)
            misclassified = np.zeros([results_matrix.shape[0], results_matrix.shape[1]])
            misclassified[np.where(results_matrix == 0)] = 1
        
            plt.figure(2, figsize=(3,5))
            SMALL_SIZE = 12
            MEDIUM_SIZE = 14
            BIGGER_SIZE = 14

            plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
            plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            plt.imshow(misclassified, cmap='Oranges', extent=[-30,30,100,0])
            plt.xlabel("Angle")
            plt.ylabel("Examples")
            ax = plt.gca()
            plt.xticks([-30, -15, 0, 15, 30])
            ax.set_yticklabels([])
            plt.yticks([])
            plt.savefig(os.path.join(
                save_plot_subdir_subdir, 
                "{}_{}_misclassified_angles_{}_k{}_{}.pdf".format(
                    dataset, split, args.method, args.k, args.exp_id)))
            plt.close()

            if config.data.dataset_name == "cifar-10":
                bins = 10
                data_iterator = cifar10_input.CIFAR10Data(config.data.data_path)
                label_names = data_iterator.label_names
            elif config.data.dataset_name == "cifar-100":
                bins = 100
                data_iterator = cifar100_input.CIFAR100Data(config.data.data_path)
                label_names = data_iterator.label_names
            elif config.data.dataset_name == "svhn":
                bins = 10
                data_iterator = svhn_input.SVHNData(config.data.data_path)
                label_names = np.arange(bins)

            # images fooled
            if fool_tup is not None:
                imgs_fooled, y_fooled, pred_fooled, imgs_f_per, pred_f_per = fool_tup
                plot_images(imgs_fooled, y_fooled, pred_fooled, imgs_f_per,
                            pred_f_per, bins, label_names, 
                            save_plot_subdir, 'misclassified_all_angles', 
                            args.method, args.exp_id, args.k)

            # images all correct
            if cor_tup is not None:
                imgs_correct, y_correct, pred_correct, imgs_c_per, pred_c_per = cor_tup
                plot_images(imgs_correct, y_correct, pred_correct, imgs_c_per, 
                            pred_c_per, bins, label_names, 
                            save_plot_subdir, 'correct_all_angles', args.method, 
                            args.exp_id, args.k)
            
            # images almost all correct
            if some_cor_tup is not None:
                (imgs_some_correct, y_some_correct, pred_some_correct, imgs_sc_per, 
                 pred_sc_per) = some_cor_tup
                plot_images(imgs_some_correct, y_some_correct, pred_some_correct, 
                            imgs_sc_per, pred_sc_per,
                            bins, label_names, save_plot_subdir, 'correct_almost_all_angles', 
                            args.method,  args.exp_id, args.k)
