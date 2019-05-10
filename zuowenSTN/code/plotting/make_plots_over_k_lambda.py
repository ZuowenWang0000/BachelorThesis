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
    parser.add_argument('--dir_base', type=str,
                        help='path to folder with baseline results',
                        default='../../draft/plots/CIFAR-10/spatial/jsons', 
                        required=False)
    parser.add_argument('--results_file_base', type=str,
                        help='path to results file for baseline',
                        default='test_nat_with_crops_cifar10.json', 
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

    base_path = os.path.join(args.dir_base, args.results_file_base)
    results_dict_base = plot_utils.load_json(base_path)

    results_dict_attack = plot_utils.load_json(
        os.path.join(args.repo_dir, args.results_file))
    
    method_legend_list = ['RECOVARIT (50) w/ Wo-k', 'Adv. Training (50) w/ Wo-k', 
        'Adv. Training (100) w/ Wo-k', 'Standard']
    method_legend_list_lam = ['RECOVARIT (50)', 'Adv. Training (50) w/ Wo-k',
        'Adv. Training (100) w/ Wo-k']
    #fmt_dict = {'CoRe': 'r,-', 'Worst-of-k (50)': 'g,-.', 
    #    'Worst-of-k (100)': 'b,--', 'Standard': 'cv--'}
    fmt_dict = {'CoRe': 'r.-', 'Worst-of-k (50)': 'g^-.',
        'Worst-of-k (100)': 'b*--', 'Standard': 'cv--'}
    accuracies = ['test_grid_accuracy', 'test_nat_accuracy']
    ylabels = ['Test grid accuracy', 'Test natural accuracy']

    print("Plotting lambda comparison for different k")
    # lambda comparison for different worst-of-k 
    for j, acc in enumerate(accuracies):
        base_y_mean, base_y_std = results_dict_base['{}_summary'.format(acc)]
        plot_utils.plot_core_vs_robust_optim_acc_lambda(
            results_dict_attack, 
            'lambda_core', 
            acc, 
            attack,
            add_baseline=True,
            baseline_y=base_y_mean, 
            fmt_dict=fmt_dict,
            dataset=dataset,
            title='', #'{}: Test accuracies for different core penalty weights'.format(dataset),
            xlabel='Penalty weight',
            ylabel=ylabels[j],
            y_legend_list=method_legend_list_lam,
            markersize=6,
            axis=[0,2,0,100],
            figsize=(5,3),
            save_to_file=True,
            save_path=save_plot_subdir, 
            save_name='{}_lambda_acc_{}.pdf'.format(dataset, acc))

    
    print("Plotting lambda comparison for different k whole range")
    # lambda comparison for different worst-of-k 
    for j, acc in enumerate(accuracies):
        base_y_mean, base_y_std = results_dict_base['{}_summary'.format(acc)]
        plot_utils.plot_core_vs_robust_optim_acc_lambda(
            results_dict_attack, 
            'lambda_core', 
            acc, 
            attack,
            add_baseline=True,
            baseline_y=base_y_mean, 
            fmt_dict=fmt_dict,
            dataset=dataset,
            title='',#'{}: Test accuracies for different core penalty weights'.format(dataset),
            xlabel='Penalty weight',
            ylabel=ylabels[j],
            y_legend_list=method_legend_list_lam,
            markersize=6,
            figsize=(5,3),
            save_to_file=True,
            save_path=save_plot_subdir, 
            save_name='{}_lambda_acc_{}_full.pdf'.format(dataset, acc))


    print("Plotting worst-of-k comparison for given core lambda")
    # worst-of-k comparison for given core lambda
    for j, acc in enumerate(accuracies):
        base_y_mean, base_y_std = results_dict_base['{}_summary'.format(acc)]
        plot_utils.plot_core_vs_robust_optim_acc(
            results_dict_attack, 'worstofk', acc, 
            select_best_lambda=True, 
            add_baseline=True,
            baseline_y=base_y_mean,
            title='', #'{}: Test accuracies for worst-of-k training'.format(dataset),
            xlabel='k',
            ylabel=ylabels[j],
            y_legend_list=method_legend_list, 
            fmt_dict=fmt_dict,
            dataset=dataset,
            markersize=6,
            figsize=(5,3),
            save_to_file=True,
            save_path=save_plot_subdir, 
            save_name='{}_worstofk_acc_{}_lambda{}.pdf'.format(
                dataset, acc, chosen_lambda))

   
