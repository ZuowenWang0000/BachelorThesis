import argparse
import copy
import json
import os
import pickle

from collections import namedtuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import plotting_fct as plot_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Extract experiment results',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--repo_dir', type=str,
                        help='path to repo folder',
                        default='../../draft/plots/CIFAR-10/spatial/jsons', 
                        required=False)
    parser.add_argument('-r', '--results_file', type=str,
                        help='path to results file',
                        default='experiments_cifar10_spatial_all.json', 
                        required=False)
    parser.add_argument('-s', '--save_folder', type=str,
                        help='path to plots folder',
                        default='../../draft/plots/', required=False)
    parser.add_argument('--dataset', type=str,
                        help='dataset name',
                        default='CIFAR-10', required=False)
    parser.add_argument('--k', type=int,
                        help='k for worst-of-k',
                        default=10, required=False)  
    parser.add_argument('--core_lambda', type=float,
                        help='k for worst-of-k',
                        default=.25, required=False)                      
    args = parser.parse_args()
    plot_training_curves = True
    worstofk = [1, 5, 10, 20]
    attack = 'spatial'
    dataset = args.dataset
    chosen_k = args.k
    chosen_lambda = 'best'

    # create plot subdirectories
    save_plot_subdir = os.path.join(args.save_folder, dataset, attack)
    os.makedirs(save_plot_subdir, exist_ok=True)

    results_dict_attack = plot_utils.load_json(
        os.path.join(args.repo_dir, args.results_file))

    # casting runtime to hours
    for key in results_dict_attack:
        results_dict_attack[key]['runtime_wo_eval_in_hours'] = (
            results_dict_attack[key]['runtime_wo_eval']/3600)
    
    method_legend_list = ['RECOVARIT (50) w/ Wo-k', 'Adv. Training (50) w/ Wo-k', 
        'Adv. Training (100) w/ Wo-k']
    # method_legend_list = ['RECOVARIT', 'Adv. Training 1', 
    #     'Adv. Training 2']
    fmt_dict = {'core': 'r.-', 'rob50': 'g^-.', 'rob100': 'b*--'}
    accuracies = ['test_grid_accuracy']
    ylabels = ['Test grid accuracy']

    print("Plotting worst-of-k comparison for given core lambda")
    # worst-of-k comparison for given core lambda
    for j, acc in enumerate(accuracies):
        plot_utils.plot_acc_vs_runtime(
            results_dict_attack, 
            'runtime_wo_eval_in_hours', 
            acc, 
            select_best_lambda=True, 
            title='', 
            xlabel='Runtime (in h)',
            ylabel=ylabels[j],
            y_legend_list=method_legend_list, 
            fmt_dict=fmt_dict,
            dataset=dataset,
            markersize=8,
            figsize=(5,5),
            save_to_file=True,
            save_path=save_plot_subdir, 
            save_name='{}_{}_runtime_detailed.pdf'.format(dataset, acc))

