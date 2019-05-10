import argparse
import json
import os
import pickle

from collections import namedtuple

import numpy as np


def load_json(path):
    with open(path) as f:
        json_contents = json.load(f)
    return json_contents



def extract_exp_id(config, filter_by_dict=None):
    matching_ids = []
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
                    matching_ids.append(k)
            else:
                matching_ids.append(k)

    return matching_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Extract experiment results',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--repo_dir', type=str,
                        help='path to repo folder',
                        default='/Users/heinzec/projects/core-da/repo_dir_7jan/', 
                        required=False)
    parser.add_argument('-r', '--results_file', type=str,
                        help='path to results file',
                        default='experiments_cifar10_spatial_all.json', 
                        required=False)
    parser.add_argument('--k', type=int,
                        help='k for worst-of-k',
                        default=10, required=False)  
    parser.add_argument('--core_lambda', type=float,
                        help='lambda for core',
                        default=.25, required=False)  
    parser.add_argument('--use_core', type=int,
                        help='flag whether to extract core results',
                        default=1, required=False)

    args = parser.parse_args()
    results_dict = load_json(os.path.join(args.repo_dir, args.results_file))
    chosen_k = args.k
    chosen_lambda = args.core_lambda
    filter_dict = {}
    filter_dict['worstofk'] = chosen_k
    filter_dict['lambda_core'] = chosen_lambda
    filter_dict['hyperparameters.training.use_core'] = args.use_core
    ids = extract_exp_id(results_dict, filter_dict)
    print(ids)
