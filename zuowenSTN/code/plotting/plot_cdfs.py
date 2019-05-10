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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Extract experiment results',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--results_folder', type=str,
                        help='path to results file',
                        default='../../draft/plots/CIFAR-100/spatial/pkl_ig', 
                        required=False)
    parser.add_argument('--save_folder', type=str,
                        help='path to plots folder',
                        default='../../draft/plots/', required=False)
    parser.add_argument('--dataset', type=str,
                        help='dataset name',
                        default='CIFAR-100', required=False)
    parser.add_argument('--attack', type=str,
                        help='attack name',
                        default='spatial', required=False)
    parser.add_argument('--k', type=int,
                        help='k for worst-of-k',
                        default=10, required=False)  
    parser.add_argument('--exp_id_list', type=str, nargs='+',
                        default=['7qQohWhPDN_1059271', 'KYkyZUnXt3_1059184',
'EsM4HpxjCU_1059266'])                   
    args = parser.parse_args()
    save_plot_subdir = os.path.join(args.save_folder, args.dataset, args.attack)
    runs_str = '_'.join([s for s in args.exp_id_list])
    save_path = os.path.join(save_plot_subdir, 
                "{}_cdf_misclassified_k{}_runs{}.pdf".format(
                    args.dataset, args.k, runs_str))

    results_folder = args.results_folder
    pkl_files_all = [os.path.join(results_folder, f) 
        for f in os.listdir(results_folder)
        if os.path.isfile(os.path.join(results_folder, f))]

    # filter pkl files
    pkl_files = [f for f in pkl_files_all if any(
        [e in f for e in args.exp_id_list])]

    row_means_dict = {}
    correct_row_means_dict = {}
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            dat = pickle.load(f)
            dat_rev = np.zeros([dat.shape[0], dat.shape[1]])
            dat_rev[np.where(dat == 0)] = 1
            if 'core' in pkl_file:
                row_means_dict['core'] = np.sum(dat, axis=1)
                correct_row_means_dict['core'] = np.sum(dat_rev, axis=1)
            elif 'rob50' in pkl_file:
                row_means_dict['rob50'] = np.sum(dat, axis=1)
                correct_row_means_dict['rob50'] = np.sum(dat_rev, axis=1)
            elif 'rob100' in pkl_file:
                row_means_dict['rob100'] = np.sum(dat, axis=1)
                correct_row_means_dict['rob100'] = np.sum(dat_rev, axis=1)
            else:
                raise NotImplementedError

    plt.figure(1, figsize=(5,3))
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
    handles = []
    labels = []
    linewidth = 2
    key_order = ['core', 'rob50', 'rob100']
    legend_label_dict = {'core': 'RECOVARIT (50) w/ Wo-k', 'rob50': 'Adv. Training (50) w/ Wo-k', 
        'rob100': 'Adv. Training (100) w/ Wo-k'}
    color_dict = {'core': 'red', 'rob50': 'green', 'rob100': 'blue'}
    lsty_dict = {'core': '-', 'rob50': '-.', 'rob100': '--'}
   
    n_grid_points = dat.shape[1]
    for key in key_order:
        dat = row_means_dict[key]
        label = legend_label_dict[key]
        labels.append(label)
        counts = [sum(dat == i) for i in range(n_grid_points+1)]
        cdf = np.cumsum(counts)
        xvals = np.linspace(0, 1, num=len(cdf))
        p, = plt.plot(xvals, cdf/cdf[-1], label=label,
                      color=color_dict[key], linestyle=lsty_dict[key],
            linewidth=linewidth)
        handles.append(p)
    xlim = .2
    if args.dataset == "SVHN":
        if args.k == 10:
            plt.axis([0,xlim,0.85,1])
        elif args.k == 1:
            plt.axis([0,xlim,0.75,1])
        else:
            plt.axis([0,xlim,0.75,1])
    elif args.dataset == "CIFAR-10":
        if args.k == 10:
            plt.axis([0,xlim,0.6,1])
        elif args.k == 1:
            plt.axis([0,xlim,0.45,1])
        else:
            plt.axis([0,xlim,0.4,1])
    elif args.dataset == "CIFAR-100":
        if args.k == 10:
            plt.axis([0,xlim,0.3,.7])
        elif args.k == 1:
            plt.axis([0,xlim,0.2,.7])
        else:
            plt.axis([0,xlim,0.2,1])
    else:
        plt.axis([0,xlim,0,1])
    plt.grid(True, linestyle=':')
    legend_position = 'upper left'
    bbox_to_anchor=(0., 1.3)
    ax = plt.gca()
    #ax.legend(handles=handles, labels=labels,
    #    loc=legend_position, ncol = 1,
    #    bbox_to_anchor=bbox_to_anchor)
    plt.legend(handles, labels, loc='lower right')
    # plt.title("{}: k = {}".format(args.dataset, args.k))
    plt.xlabel("Fraction of misclassified transformations")
    plt.ylabel("Cumulative density")
    ax = plt.gca()
    plot_utils.adjust_spines(ax, ['left', 'bottom'])
    plt.savefig(save_path, pad_inches=.1, bbox_inches='tight')

    # plt.figure(2)
    # handles = []
    # labels = []
    # linewidth = 2
    # legend_label_dict = {'core': 'CoRe', 'rob50': 'Worst-of-k (50)', 
    #     'rob100': 'Worst-of-k (100)'}
    # color_dict = {'CoRe': 'red', 'Worst-of-k (50)': 'green', 'Worst-of-k (100)': 'blue'}
    # lsty_dict = {'CoRe': '--', 'Worst-of-k (50)': '-.', 'Worst-of-k (100)': ':'}
    # for key in row_means_dict:
    #     dat_rev = correct_row_means_dict[key]
    #     label = legend_label_dict[key]
    #     labels.append(label)
    #     num_bins = 100
    #     counts, bin_edges = np.histogram(dat_rev, bins=num_bins, normed=True)
    #     cdf = np.cumsum(counts)
    #     p, = plt.plot(bin_edges[1:], cdf/cdf[-1], label=label,
    #         color=color_dict[label], linestyle=lsty_dict[label],
    #         linewidth=linewidth)
    #     handles.append(p)
    # plt.axis([0,1,0,1])
    # plt.legend(handles, labels, loc='lower right')
    # plt.title("")
    # plt.xlabel("Fraction of correctly classified transformations")
    # plt.ylabel("Cumulative density")
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tick_params(top='off', right='off')
    # plt.show()
