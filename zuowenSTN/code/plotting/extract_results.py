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
                        default='/Users/heinzec/projects/core-da/repo_dir_7jan/', 
                        # default='/Users/heinzec/Desktop/', 
                        required=False)
    parser.add_argument('-r', '--results_file', type=str,
                        help='path to results file',
                        default='experiments_cifar10_spatial_all.json', 
                        # default='experiments_cifar100_spatial_160k.json', 
                        required=False)
    parser.add_argument('-m', '--metadata_folder', type=str,
                        help='folder name containing training metadata',
                        default='training_metadata', required=False)
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
    plot_training_curves = False
    worstofk = [1, 5, 10, 20]
    
    results_dict = plot_utils.load_json(
        os.path.join(args.repo_dir, args.results_file))
    # casting runtime to hours
    for key in results_dict:
        results_dict[key]['runtime_wo_eval_in_hours'] = (
            results_dict[key]['runtime_wo_eval']/3600)

    attack = 'spatial'
    method_legend_list = ['RECOVARIT (50) w/ Wo-k', 'Adv. Training (50) w/ Wo-k', 
        'Adv. Training (100) w/ Wo-k']
    fmt_dict = {'CoRe': 'r.-', 'Worst-of-k (50)': 'g^-.', 'Worst-of-k (100)': 'b*--'}
    dataset = args.dataset
    chosen_k = args.k
    chosen_lambda = args.core_lambda

    # create plot subdirectories
    save_plot_subdir = os.path.join(args.save_folder, dataset, attack)
    os.makedirs(save_plot_subdir, exist_ok=True)

    accuracies = ['test_grid_accuracy', 'test_adv_accuracy', 'test_nat_accuracy']
    ylabels = ['Test grid accuracy', 'Test worst-of-1 accuracy', 'Test natural accuracy']

    # print("Plotting lambda comparison for different k")
    # # lambda comparison for different worst-of-k 
    # for j, acc in enumerate(accuracies):
    #     plot_utils.plot_core_vs_robust_optim_acc_lambda(
    #         results_dict, 
    #         'lambda_core', 
    #         acc, 
    #         attack, 
    #         title='{}: Test accuracies for different core penalty weights'.format(dataset),
    #         xlabel='weight',
    #         ylabel=ylabels[j],
    #         save_to_file=True,
    #         save_path=save_plot_subdir, 
    #         save_name='{}_lambda_acc_{}.pdf'.format(dataset, acc))

    # print("Plotting worst-of-k comparison for given core lambda")
    # # worst-of-k comparison for given core lambda
    # for j, acc in enumerate(accuracies):
    #     filter_dict_core = {}
    #     # TODO: pick best lambda
    #     filter_dict_core['lambda_core'] = chosen_lambda
    #     plot_utils.plot_core_vs_robust_optim_acc(results_dict, 'worstofk', acc, 
    #         filter_dict_core, 
    #         title='{}: Test accuracies for worst-of-k training'.format(dataset),
    #         xlabel='k',
    #         ylabel=ylabels[j],
    #         y_legend_list=method_legend_list, 
    #         fmt_dict=fmt_dict,
    #         save_to_file=True,
    #         save_path=save_plot_subdir, 
    #         save_name='{}_worstofk_acc_{}_lambda{}.pdf'.format(
    #             dataset, acc, chosen_lambda))

    # runtime
    print("Plotting runtime")
    filter_dict_core = {}
    filter_dict_core['lambda_core'] = chosen_lambda
    plot_utils.plot_core_vs_robust_optim_acc(
        results_dict, 'worstofk', 'runtime_wo_eval_in_hours',
        filter_dict_core, 
        title='', #{}: Runtime'.format(dataset),
        xlabel='k',
        ylabel='Runtime (in hours)',
        y_legend_list=method_legend_list, 
        fmt_dict=fmt_dict,
        markersize=6,
        figsize=(5,3),
        save_to_file=True,
        save_path=save_plot_subdir, 
        save_name='{}_runtime_lambda{}.pdf'.format(dataset, chosen_lambda))

    if plot_training_curves:
        # training curves
        y_par_list=['test_nat_accuracy', 'test_adv_accuracy', 
            'train_inp_accuracy', 'train_adv_accuracy', 'train_nat_accuracy']
        y_labels_list = ['Standard test accuracy', 'Wo1 test accuracy', 
            'Input train accuracy', 'Wo1 train accuracy', 'Standard train accuracy']

        for j, ypar in enumerate(y_par_list):
            filter_dict_core = {}
            filter_dict_core['lambda_core'] = chosen_lambda
            filter_dict_core['worstofk'] = chosen_k
            filter_dict_robust_optim_100 = {}
            filter_dict_robust_optim_100['worstofk'] = chosen_k
            filter_dict_robust_optim_50 = {}
            filter_dict_robust_optim_50['worstofk'] = chosen_k
            plot_utils.plot_training_curves_different_methods(
                results_dict=results_dict, 
                folder_path=os.path.join(args.repo_dir, args.metadata_folder), 
                filter_dict_core=filter_dict_core, 
                filter_dict_robust_optim_50=filter_dict_robust_optim_50,
                filter_dict_robust_optim_100=filter_dict_robust_optim_100,
                y_par=ypar, 
                fmt_dict=fmt_dict, 
                title='{}: Training curves'.format(dataset), 
                xlabel='Training steps', 
                ylabel=y_labels_list[j], 
                y_legend_list=method_legend_list, 
                save_to_file=True,
                save_path=save_plot_subdir,
                save_name='{}_training_curves_{}_lambda{}_k{}.pdf'.format(
                    dataset, ypar, chosen_lambda, chosen_k))

        for method in method_legend_list:
            filter_dict = {}
            filter_dict['worstofk'] = chosen_k
            if method == 'CoRe':
                filter_dict['lambda_core'] = chosen_lambda
                filter_dict['hyperparameters.training.use_core'] = 1
            elif method == 'Worst-of-k (50)':
                filter_dict['hyperparameters.training.use_core'] = 1
                filter_dict['lambda_core'] = 0
            elif method == 'Worst-of-k (100)':
                filter_dict['hyperparameters.training.use_core'] = 0
            else:
                raise NotImplementedError
            
            plot_utils.plot_training_curves_different_ypars(
                results_dict=results_dict, 
                folder_path=os.path.join(args.repo_dir, args.metadata_folder), 
                filter_dict=filter_dict, 
                y_par_list=y_par_list, 
                y_legend_list=y_labels_list,
                fmt_list=['g.-', 'c.-', 'r.-', 'b.-', 'y.-'], 
                title='{}: Training curves for {}'.format(dataset, method), 
                xlabel='Training steps', 
                ylabel='Accuracy', 
                save_to_file=True,
                save_path=save_plot_subdir,
                save_name='{}_training_curves_{}_lambda{}_k{}.pdf'.format(
                    dataset, method, chosen_lambda, chosen_k))
