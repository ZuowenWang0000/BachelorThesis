import argparse
import json
import os
import pickle

from collections import namedtuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import plotting_fct as plot_utils


def extract_single_metric(config, metric, filter_by_dict=None):
    results_dict = {}
    
    for k in config:
        if config[k]['completed']:
            if filter_by_dict != None:
                match = True
                # filter out runs not matching the filter
                for key in filter_by_dict:
                    nested_keys = str.split(key, '.')
                    value = config[k]
                    for nk in nested_keys:
                        value = value[nk]
                    if value != filter_by_dict[key]:
                        match = False
                # config needs to match all entries in filter dict 
                if match:
                    results_dict[k] = config[k][metric]
            else:
                results_dict[k] = config[k][metric]
    results_list = [results_dict[k] for k in results_dict]
    # compute summaries over runs
    print('Averaging results from {} runs'.format(len(results_list)))
    mean = np.mean(results_list)
    std = np.std(results_list)
    return {metric: (mean, std)}



def extract_metrics(config, metrics, filter_by_dict=None):
    results_dict = {}
    
    for k in config:
        if config[k]['completed']:
            if filter_by_dict != None:
                match = True
                # filter out runs not matching the filter
                for key in filter_by_dict:
                    nested_keys = str.split(key, '.')
                    value = config[k]
                    for nk in nested_keys:
                        value = value[nk]
                    if value != filter_by_dict[key]:
                        match = False
                # config needs to match all entries in filter dict 
                if match:
                    for metric in metrics:
                        if k not in results_dict:
                            results_dict[k] = {}
                        results_dict[k][metric] = config[k][metric]
            else:
                for metric in metrics:
                    if k not in results_dict:
                        results_dict[k] = {}
                    results_dict[k][metric] = config[k][metric]
            
    out_dict = {}
    for metric in metrics:             
        results_list = [results_dict[k][metric] for k in results_dict]
        # compute summaries over runs
        print('Averaging results from {} runs'.format(len(results_list)))
        mean = np.mean(results_list)
        std = np.std(results_list)
        out_dict[metric] = (mean, std)
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Extract experiment results',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--repo_dir', type=str,
                        help='path to repo folder',
                        default='/Users/heinzec/Desktop/', 
                        required=False)
    parser.add_argument('-r', '--results_file', type=str,
                        help='path to results file',
                        default='experiments_cifar100_linf_all.json', 
                        required=False)
    parser.add_argument('-s', '--save_folder', type=str,
                        help='path to plots folder',
                        default='../../draft/plots/', required=False)
    parser.add_argument('--dataset', type=str,
                        help='dataset name',
                        default='CIFAR-100', required=False)
    parser.add_argument('--attack', type=str,
                        help='attack name',
                        default='linf', required=False)
    parser.add_argument('--metrics', type=str,  nargs='+',
                        help='Metrics to extract',
                        default=['test_adv_accuracy', 'test_nat_accuracy'], 
                        required=False)      
    parser.add_argument('--epsilon', type=int, nargs='+',
                        help='eps L-inf',
                        default=[16], required=False)  
    parser.add_argument('--core_lambdas', type=float, nargs='+',
                        help='lambdas for core',
                        default=[.25, .5, .75, 1, 1.5, 2], required=False)        
    parser.add_argument('--methods', type=str,  nargs='+',
                        help='Methods',
                        default=['core', 'rob50', 'rob100'], required=False)   
    parser.add_argument('--save_filename', type=str,
                        help='path to plots folder',
                        default='averages_core_robust_linf.json', required=False)   
    parser.add_argument('--save', type=int, 
                        help='flag whether to save results',
                        default=1, required=False)                                                                                                            
    args = parser.parse_args()
    dataset = args.dataset
    attack = args.attack
    results_dict = plot_utils.load_json(
        os.path.join(args.repo_dir, args.results_file))
    # create plot subdirectories
    save_plot_subdir = os.path.join(args.save_folder, dataset, attack)
    os.makedirs(save_plot_subdir, exist_ok=True)

    save_filename = os.path.join(save_plot_subdir, 
        '{}_{}_{}'.format(dataset, attack, args.save_filename))

    out_dict = {}
    for method in args.methods:
        results = {}
        filter_dict = {}

        if method == 'nat':
            filter_dict['hyperparameters.training.use_core'] = 0
            key_k = "eps: {}".format(0)
            results[key_k] = {}
            key_lam = "core_lambda: {}".format(0)
            results[key_k][key_lam] = {}
            results[key_k][key_lam]['hyperparameters.training.use_core'] = 0
            results_metric = extract_metrics(results_dict, args.metrics, 
                filter_by_dict=filter_dict)
            results[key_k][key_lam].update(results_metric) 
        else:
            for eps in args.epsilon:
                key_k = "eps: {}".format(eps)
                filter_dict['hyperparameters.attack.epsilon'] = eps
                results[key_k] = {}

                if method == 'core':
                    for lam in args.core_lambdas:
                        key_lam = "core_lambda: {}".format(lam)
                        filter_dict['hyperparameters.training.use_core'] = 1
                        filter_dict['lambda_core'] = lam
                        results[key_k][key_lam] = {}
                        results[key_k][key_lam]['hyperparameters.training.use_core'] = 1
                        results_metric = extract_metrics(results_dict, args.metrics, 
                            filter_by_dict=filter_dict)
                        results[key_k][key_lam].update(results_metric) 
                    
                else:
                    key_lam = "core_lambda: {}".format(0)
                    if method == 'rob50':
                        filter_dict['hyperparameters.training.use_core'] = 1
                        filter_dict['lambda_core'] = 0
                        results[key_k][key_lam] = {}
                        results[key_k][key_lam]['hyperparameters.training.use_core'] = 1
                    elif method == 'rob100':
                        filter_dict['lambda_core'] = 0
                        filter_dict['hyperparameters.training.use_core'] = 0
                        results[key_k][key_lam] = {}
                        results[key_k][key_lam]['hyperparameters.training.use_core'] = 0                    
                    else:
                        raise NotImplementedError

                    results_metric = extract_metrics(results_dict, args.metrics, 
                        filter_by_dict=filter_dict)
                    results[key_k][key_lam].update(results_metric)     
        out_dict[method] = results
        
    # save results
    with open(save_filename, 'w') as result_file:
        json.dump(out_dict, result_file, sort_keys=True, indent=4)
