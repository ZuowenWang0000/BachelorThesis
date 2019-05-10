import argparse
import json
import os
import pickle

from collections import namedtuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np


def load_json(path):
    with open(path) as f:
        json_contents = json.load(f)
    return json_contents


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
                        default='test_core_lam025_k10_finer_grid_test_acc.json', 
                        required=False)
    parser.add_argument('--split', type=str,
                        help='train or test split',
                        default='test', 
                        required=False)
    args = parser.parse_args()
    res_fname = os.path.join(args.repo_dir, args.results_file)
    results_dict = load_json(res_fname)
    test_grid_accuracy = []
    test_nat_accuracy = []
    for key in results_dict:
        if key != 'hyperparameters':
            test_grid_accuracy.append(results_dict[key]['{}_grid_accuracy'.format(args.split)])
            test_nat_accuracy.append(results_dict[key]['{}_nat_accuracy'.format(args.split)])
    
    results_dict['{}_grid_accuracy_summary'.format(args.split)] = (
        np.mean(test_grid_accuracy), np.std(test_grid_accuracy))
    results_dict['{}_nat_accuracy_summary'.format(args.split)] = (
        np.mean(test_nat_accuracy), np.std(test_nat_accuracy))
    
    with open(res_fname, 'w') as result_file:
        json.dump(results_dict, result_file, sort_keys=True, indent=4)
