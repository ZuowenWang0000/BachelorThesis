import argparse
import copy
import json
import os
import pickle

from collections import namedtuple

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import container

import numpy as np

def load_json(path):
    with open(path) as f:
        json_contents = json.load(f)
    return json_contents


def get_axis_range(result, lowerx=-1, upperx=-1, lowery=-1, uppery=-1, 
                   add_margin_upper=True, margin_factor_upper=.05, 
                   add_margin_lower=False, margin_factor_lower=.1):
    if lowerx != -1:
        lowerx_ = lowerx
    else:
        lowerx_ = min(result[0])
    if lowery != -1:
        lowery_ = lowery
    else:
        lowery_ = min(result[1])
    if upperx != -1:
        upperx_ = upperx
    else:
        upperx_ = max(result[0])
    if uppery != -1:
        uppery_ = uppery
    else:
        uppery_ = max(result[1])
    if add_margin_upper:
        uppery_ = uppery_*(1+margin_factor_upper)
        upperx_ = upperx_*(1+margin_factor_upper)
    if add_margin_lower:
        lowery_ = lowery_*(1-margin_factor_lower)
        lowerx_ = lowerx_*(1-margin_factor_lower)
    return [lowerx_, upperx_, lowery_, uppery_]


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(False)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


def get_axis_range_multiple(results, lowerx=-1, upperx=-1, lowery=-1, uppery=-1,
                            add_margin_upper=True, margin_factor_upper=.05, 
                            add_margin_lower=False, margin_factor_lower=.1):
    min_x = min([min(result[0]) for result in results if len(result[0]) != 0])
    max_x = max([max(result[0]) for result in results if len(result[0]) != 0])
    min_y = min([min(result[1]) for result in results if len(result[1]) != 0])
    max_y = max([max(result[1]) for result in results if len(result[1]) != 0])

    if lowerx != -1:
        lowerx_ = lowerx
    else:
        lowerx_ = min_x
    if lowery != -1:
        lowery_ = lowery
    else:
        lowery_ = min_y
    if upperx != -1:
        upperx_ = upperx
    else:
        upperx_ = max_x
    if uppery != -1:
        uppery_ = uppery
    else:
        uppery_ = max_y
    if add_margin_upper:
        uppery_ = uppery_*(1+margin_factor_upper)
        upperx_ = upperx_*(1+margin_factor_upper)
    if add_margin_lower:
        lowery_ = lowery_*(1-margin_factor_lower)
        lowerx_ = lowerx_*(1-margin_factor_lower)
    return [lowerx_, upperx_, lowery_, uppery_]    


def get_unique_xticks(results):
    unique_xvals_all = [np.unique(result[0]) for result in results 
        if len(result[0]) != 0]
    flattened_xvals = [item for sublist in unique_xvals_all for item in sublist]
    return np.unique(flattened_xvals)


def get_metadata(metadata_dict, metadata_folder, results_dict_k, px, py):
    metadata_k = []
    with open(
        os.path.join(
            metadata_folder, results_dict_k['id']+'_training_metadata.pkl'), 
        'rb') as f:
        training_metadata = pickle.load(f)
    checkpoints = training_metadata['checkpoints']
    for k in checkpoints:
        p_x_ = checkpoints[k][px]
        p_y_ = checkpoints[k][py]
        metadata_k.append([p_x_, p_y_])
    # sort results by parameter_x
    metadata_k = sorted(metadata_k, key=lambda x: x[0])
    for x, y in metadata_k:
        if x in metadata_dict:
            metadata_dict[x].append(y)
        else:
            metadata_dict[x] = [y]
    return metadata_dict


def get_results(results_dict, configk, px, py):
    p_x_ = configk[px]
    p_y_ = configk[py]
    if p_x_ in results_dict:
        results_dict[p_x_].append(p_y_)
    else:
        results_dict[p_x_] = [p_y_]
    return results_dict


def get_results_with_selection(results_dict, configk, px, py, ps, ms):
    p_x_ = configk[px]
    p_y_dict = {py: configk[py], ps: configk[ps], ms: configk[ms]}
    if p_x_ in results_dict:
        results_dict[p_x_].append(p_y_dict)
    else:
        results_dict[p_x_] = [p_y_dict]
    return results_dict


def extract_metadata(results_dict, metadata_folder, parameter_x, parameter_y,
                      filter_by_dict, average=True):
    metadata_dict = {}
    
    for k in results_dict:
        if results_dict[k]['completed']:
            if filter_by_dict != None:
                match = True
                # filter out runs not matching the filter
                for key in filter_by_dict:
                    nested_keys = str.split(key, '.')
                    value = results_dict[k]
                    for nk in nested_keys:
                        value = value[nk]
                    if value != filter_by_dict[key]:
                        match = False
                if match:        
                    metadata_dict = get_metadata(metadata_dict, metadata_folder, 
                                                results_dict[k], parameter_x, 
                                                parameter_y)
                                                
            else:
                metadata_dict = get_metadata(metadata_dict, metadata_folder, 
                                            results_dict[k], parameter_x, 
                                            parameter_y)
    if average:
        # compute summaries over runs
        metadata_summaries = {}
        for key in metadata_dict:
            print('Averaging results from {} runs'.format(len(metadata_dict[key])))
            mean = np.mean(metadata_dict[key])
            std = np.std(metadata_dict[key])
            metadata_summaries[key] = (mean, std)

        # output as list containing three lists 
        x_values = []
        ymean_values = []
        ystd_values = []
        for key in sorted(metadata_summaries.keys()):
            ymean, ystd = metadata_summaries[key]
            x_values.append(key)
            ymean_values.append(ymean)
            ystd_values.append(ystd)

        result = []
        result.append(x_values)
        result.append(ymean_values)
        result.append(ystd_values)
    else:
        xvals = sorted(metadata_dict.keys())
        result = []
        yvals = {}
        for key in sorted(metadata_dict.keys()):
            for run_j, v in enumerate(metadata_dict[key]):
                if run_j in yvals:
                    yvals[run_j].append(v)
                else:
                    yvals[run_j] = [v]
        for key in sorted(yvals.keys()):
            result.append((xvals, yvals[key]))
    return result


def extract_data(config, parameter_x, parameter_y, filter_by_dict=None, 
        average=True):
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
                    results_dict = get_results(
                        results_dict, config[k], parameter_x, parameter_y)
            else:
                results_dict = get_results(
                    results_dict, config[k], parameter_x, parameter_y)
    if average:
        # compute summaries over runs
        summaries = {}
        for key in results_dict:
            print('Averaging results from {} runs'.format(len(results_dict[key])))
            mean = np.mean(results_dict[key])
            std = np.std(results_dict[key])
            summaries[key] = (mean, std)

        # output as list containing three lists 
        x_values = []
        ymean_values = []
        ystd_values = []
        for key in sorted(summaries.keys()):
            ymean, ystd = summaries[key]
            x_values.append(key)
            ymean_values.append(ymean)
            ystd_values.append(ystd)

        result = []
        result.append(x_values)
        result.append(ymean_values)
        result.append(ystd_values)
    else:
        x_values = []
        y_values = []
        for key in sorted(results_dict.keys()):
            y = results_dict[key]
            x_values.append(key)
            y_values.extend(y)

        result = []
        result.append(x_values)
        result.append(y_values)

    return result


def extract_data_select_best(config, parameter_x, parameter_y, 
        parameter_select, metric_select, filter_by_dict=None):
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
                    results_dict = get_results_with_selection(
                        results_dict, config[k], parameter_x, parameter_y, 
                        parameter_select, metric_select)
            else:
                results_dict = get_results_with_selection(
                    results_dict, config[k], parameter_x, parameter_y, 
                    parameter_select, metric_select)
    
    results_dict_selected = {}
    best_par = {}
    for key in results_dict:
        res_list_k = results_dict[key]
        ps_vals = [dic[parameter_select] for dic in res_list_k]
        ps_vals_dict = {}
        for ps_val in np.unique(ps_vals):
            ms_vals = [dic[metric_select] for dic in res_list_k 
                if dic[parameter_select] == ps_val]
            ps_vals_dict[np.mean(ms_vals)] = ps_val
        
        best_ps = ps_vals_dict[max(ps_vals_dict.keys())]
        y_vals = [dic[parameter_y] for dic in res_list_k 
            if dic[parameter_select] == best_ps]
        results_dict_selected[key] = y_vals
        best_par[key] = best_ps

    # compute summaries over runs
    summaries = {}
    for key in results_dict_selected:
        print('Averaging results from {} runs'.format(
            len(results_dict_selected[key])))
        mean = np.mean(results_dict_selected[key])
        std = np.std(results_dict_selected[key])
        summaries[key] = (mean, std)

    # output as list containing three lists 
    x_values = []
    ymean_values = []
    ystd_values = []
    for key in sorted(summaries.keys()):
        ymean, ystd = summaries[key]
        x_values.append(key)
        ymean_values.append(ymean)
        ystd_values.append(ystd)

    result = []
    result.append(x_values)
    result.append(ymean_values)
    result.append(ystd_values)

    return result, best_par


def plot_single(data, title, xlable, ylable, axis, save_to_file=False, 
                save_path='.', save_name='training_curves.pdf'):
    x = data[0]
    y = data[1]
    std = data[2]
    plt.figure(1)
    plt.title(title)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.errorbar(x, y, yerr=std)
    plt.axis(axis)
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name))
        plt.close()
    else:
        plt.show()


def plot_training_curves_different_ypars(results_dict, 
                                         folder_path, 
                                         y_par_list, 
                                         y_legend_list, 
                                         fmt_list, 
                                         filter_dict = {}, 
                                         average=False,
                                         title = "", 
                                         xlabel = "", 
                                         ylabel = "",
                                         axis=None, 
                                         grid=True,
                                         save_to_file=False, 
                                         save_path='.', 
                                         save_name='training_curves.pdf'):
    linewidth = 1
    alpha=0.5					 					 
    plt.figure(2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    p_list = []					 
    for k, ypar in enumerate(y_par_list):
        results_ypar = extract_metadata(
            results_dict, folder_path, 'training_step', ypar, filter_dict, 
            average=average)
        if average:
            p, = plt.plot(results_ypar[0], results_ypar[1], fmt_list[k], 
                linewidth=linewidth, alpha=alpha, label=y_legend_list[k])
            if axis is None:
                axis = get_axis_range(results_ypar, lowery=0, uppery=100)
        else:
            for xv, yv in results_ypar:
                p, = plt.plot(xv, yv, fmt_list[k], linewidth=linewidth,
                    alpha=alpha, label=y_legend_list[k])
            if axis is None and len(results_ypar) > 0:
                axis = get_axis_range([xv, yv], lowery=0, uppery=100)
        print(len(results_ypar))
        if len(results_ypar) > 0:
            p_list.append(p)
    
    if axis is not None:
        plt.axis(axis)
    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    plt.legend(handles=p_list, loc='lower right')
    plt.grid(grid, linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_training_curves_different_methods(results_dict, 
                         folder_path, 
                         y_par, 
                         filter_dict_core = {}, 
                         filter_dict_robust_optim_100 = {}, 
                         filter_dict_robust_optim_50 = {}, 
                         y_legend_list=['CoRe', 'Worst-of-k (50)', 'Worst-of-k (100)'], 
                         fmt_dict={'CoRe': 'r.-', 'Worst-of-k (50)': 'g,-.', 'Worst-of-k (100)': 'bo--'},
                         average=False,
                         title = "", 
                         xlabel = "", 
                         ylabel = "",
                         axis=None, 
                         grid=True,
                         save_to_file=False, 
                         save_path='.', 
                         save_name='training_curves.pdf'):
    linewidth = 1
    alpha=0.5					 
    plt.figure(2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    list_legend_handles = []
    results_ypar = extract_metadata(
        results_dict, folder_path, 'training_step', y_par, filter_dict_core, 
        average=average)
    if average:
        p_core, = plt.plot(results_ypar[0], results_ypar[1], fmt_dict['CoRe'], 
            linewidth=linewidth, alpha=alpha, label=y_legend_list[0])
    else:
        for xv, yv in results_ypar:
            p_core, = plt.plot(xv, yv, fmt_dict['CoRe'], linewidth=linewidth, 
                alpha=alpha, label=y_legend_list[0])
    
    if len(results_ypar) > 0:
        list_legend_handles.append(p_core)

    # robust optim 50 AT results
    filter_dict_robust_optim_50['hyperparameters.training.use_core'] = 1
    filter_dict_robust_optim_50['lambda_core'] = 0
    results_ypar = extract_metadata(
        results_dict, folder_path, 'training_step', y_par, 
        filter_dict_robust_optim_50, average=average)    
    if average:
        p_50, = plt.plot(results_ypar[0], results_ypar[1], 
            fmt_dict['Worst-of-k (50)'], 
            linewidth=linewidth, alpha=alpha, label=y_legend_list[1])
    else:
        for xv, yv in results_ypar:
            p_50, = plt.plot(xv, yv, fmt_dict['Worst-of-k (50)'], 
                linewidth=linewidth, 
                alpha=alpha, label=y_legend_list[1])
    
    if len(results_ypar) > 0:
        list_legend_handles.append(p_50)

    # robust optim 100 AT results
    filter_dict_robust_optim_100['hyperparameters.training.use_core'] = 0
    results_ypar = extract_metadata(
        results_dict, folder_path, 'training_step', y_par, 
        filter_dict_robust_optim_100, average=average)
    if average:
        p_100, = plt.plot(results_ypar[0], results_ypar[1], 
            fmt_dict['Worst-of-k (100)'], 
            linewidth=linewidth, alpha=alpha, label=y_legend_list[2])
        if axis is None:
            axis = get_axis_range(results_ypar, lowery=0, uppery=100)
    else:
        for xv, yv in results_ypar:
            p_100, = plt.plot(xv, yv, fmt_dict['Worst-of-k (100)'],
                linewidth=linewidth, 
                alpha=alpha, label=y_legend_list[2])
        if axis is None:
            axis = get_axis_range([xv, yv], lowery=0, uppery=100)

    if len(results_ypar) > 0:
        list_legend_handles.append(p_100)

    # finish plot
    plt.axis(axis)
    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    if y_par == 'test_grid_accuracy':
        legend_position = 'upper center'
    elif y_par == 'test_nat_accuracy':
        legend_position = 'lower right'  
    plt.legend(handles=list_legend_handles, loc=legend_position)
    plt.grid(grid, linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_core_vs_robust_optim_acc(results_dict, 
                                  x_par, y_par,
                                  filter_dict_core = {}, 
                                  filter_dict_robust_optim_50 = {}, 
                                  filter_dict_robust_optim_100 = {}, 
                                  select_best_lambda=False,
                                  select_best_lambda_metric='test_grid_accuracy',
                                  add_baseline=False,
                                  baseline_y=None,
                                  title = "", 
                                  xlabel = "", 
                                  ylabel = "",
                                  y_legend_list=['CoRe', 'Worst-of-k (50)', 'Worst-of-k (100)'], 
                                  fmt_dict={'CoRe': 'r.-', 'Worst-of-k (50)': 'g,-.', 'Worst-of-k (100)': 'bo--'},
                                  dataset=None,
                                  linewidth=1,
                                  markersize=4,
                                  axis=None, 
                                  grid=True,
                                  figsize=(8,6),
                                  save_to_file=False, 
                                  save_path='.',  
                                  save_name='accuracies.pdf'):
    y_legend_list_local = copy.deepcopy(y_legend_list)  
    list_legend_handles = []    
    
    # core results
    filter_dict_core['hyperparameters.training.use_core'] = 1
    if select_best_lambda:
        results_core, sel_lam = extract_data_select_best(results_dict, x_par, y_par, 
            'lambda_core', select_best_lambda_metric,
            filter_by_dict=filter_dict_core)
        print('Selected lambdas')
        print(sel_lam)
    else:
        results_core = extract_data(results_dict, x_par, y_par, 
            filter_by_dict=filter_dict_core)
    
    if len(results_core[0]) == 0:
        y_legend_list_local.remove('CoRe')

    # robust optim 100 AT results
    filter_dict_robust_optim_100['hyperparameters.training.use_core'] = 0
    results_robust_100AT = extract_data(results_dict, x_par, y_par,
        filter_by_dict=filter_dict_robust_optim_100)
    if len(results_robust_100AT[0]) == 0:
        y_legend_list_local.remove('Worst-of-k (100)')

    # robust optim 50 AT results
    filter_dict_robust_optim_50['hyperparameters.training.use_core'] = 1
    filter_dict_robust_optim_50['lambda_core'] = 0
    results_robust_50AT = extract_data(results_dict, x_par, y_par,
        filter_by_dict=filter_dict_robust_optim_50)
    if len(results_robust_50AT[0]) == 0:
        y_legend_list_local.remove('Worst-of-k (50)')

    if axis is None:
        if y_par == 'runtime_wo_eval' or y_par == 'runtime_wo_eval_in_hours':
            axis = get_axis_range_multiple(
                [results_robust_100AT, results_robust_50AT, results_core],
                add_margin_lower=True, margin_factor_upper=.1, 
                margin_factor_lower=.2)
        else:
            axis = get_axis_range_multiple(
                [results_robust_100AT, results_robust_50AT, results_core],
                lowery=0, uppery=100)
    xvals = get_unique_xticks(
        [results_robust_100AT, results_robust_50AT, results_core])
    plt.figure(2, figsize=figsize)
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
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)			
    p_core = plt.errorbar(results_core[0], results_core[1], yerr=results_core[2], 
        fmt=fmt_dict['CoRe'],
        linewidth=linewidth, markersize=markersize)
    list_legend_handles.append(p_core)
    p_50 = plt.errorbar(results_robust_50AT[0], results_robust_50AT[1], 
        yerr=results_robust_50AT[2], 
        fmt=fmt_dict['Worst-of-k (50)'],
        linewidth=linewidth, markersize=markersize)
    list_legend_handles.append(p_50)
    p_100 = plt.errorbar(results_robust_100AT[0], results_robust_100AT[1], 
        yerr=results_robust_100AT[2], 
        fmt=fmt_dict['Worst-of-k (100)'],
        linewidth=linewidth, markersize=markersize)
    list_legend_handles.append(p_100)

    if y_par == 'test_grid_accuracy':
        legend_position = 'upper left'
        bbox_to_anchor=(0., 1.4)
    elif y_par == 'runtime_wo_eval_in_hours':
        legend_position = 'upper left'
        bbox_to_anchor=(0., 1.2)
    elif y_par == 'test_nat_accuracy':
        legend_position = 'upper left'
        bbox_to_anchor=(0., 1.3)

    if add_baseline:
        p_base, = plt.plot(xvals, len(xvals)*[baseline_y], 
            fmt_dict['Standard'], linewidth=linewidth, markersize=markersize)
        list_legend_handles.append(p_base)
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h 
            for h in list_legend_handles]
        ax = plt.gca()
        ax.legend(handles=handles, labels=y_legend_list,
            loc=legend_position, ncol = 1,
            bbox_to_anchor=bbox_to_anchor)
    else:
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h 
            for h in list_legend_handles]
        plt.legend(handles, y_legend_list, loc=legend_position, 
            bbox_to_anchor=bbox_to_anchor)

    if y_par == 'test_nat_accuracy':
        if dataset == "SVHN":
            axis[2] = 90 
            axis[3] = 100
        elif dataset == "CIFAR-10":
            axis[2] = 80
            axis[3] = 100
        elif dataset == "CIFAR-100":
            axis[2] = 50
            axis[3] = 85

    if y_par == 'test_grid_accuracy':
        if dataset == "CIFAR-100":
            axis[2] = 0
            axis[3] = 65
        elif dataset == "CIFAR-10":
            axis[2] = 0
            axis[3] = 90
            
    plt.axis(axis)


    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    plt.xticks(xvals)
    plt.grid(grid, linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_core_vs_robust_optim_acc_lambda(results_dict, 
                                         x_par, 
                                         y_par,
                                         attack,
                                         worstofk=[1, 5, 10, 20],
                                         filter_dict_core={}, 
                                         filter_dict_robust_optim_100={}, 
                                         filter_dict_robust_optim_50={}, 
                                         plot_robust=False,
                                         add_baseline=False,
                                         baseline_y=None,
                                         dataset=None,
                                         fmt_dict={'CoRe': 'r.-', 'Worst-of-k (50)': 'g,-.', 'Worst-of-k (100)': 'bo--'},
                                         y_legend_list=['CoRe', 'Worst-of-k (50)', 'Worst-of-k (100)'], 
                                         title="", 
                                         xlabel="", 
                                         ylabel="",
                                         lsty_dict = {1: 'b.-', 5: 'm^-.', 10: 'g*:', 20: 'rx--'},
                                         linewidth=1,
                                         markersize=4,
                                         axis=None, 
                                         grid=True,
                                         figsize=(8,6),
                                         save_to_file=False, 
                                         save_path='.',  
                                         save_name='accuracies.pdf'):
    
    plt.figure(2, figsize=figsize)
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
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)	
    list_legend_handles = []    

    # core results
    filter_dict_core['hyperparameters.training.use_core'] = 1
    if attack == 'spatial':
        for _, k in enumerate(worstofk):
            filter_dict_core['worstofk'] = k
            results_core = extract_data(
                results_dict, x_par, y_par, filter_by_dict=filter_dict_core)
            print(results_core)
            p = plt.errorbar(results_core[0], results_core[1], 
                yerr=results_core[2], fmt=lsty_dict[k],
                #linestyle=lsty_dict[k], 
                label = '{} w/ Wo-{}'.format(y_legend_list[0], k))
            list_legend_handles.append(p)
            xvals = get_unique_xticks([results_core])
            if plot_robust:
                # robust optim 100 AT results
                filter_dict_robust_optim_100['hyperparameters.training.use_core'] = 0
                filter_dict_robust_optim_100[
                    'hyperparameters.training.adversarial_training'] = True
                filter_dict_robust_optim_100['worstofk'] = k
                results_robust_100AT = extract_data(results_dict, x_par, y_par,
                    filter_by_dict=filter_dict_robust_optim_100)

                # robust optim 50 AT results
                filter_dict_robust_optim_50['hyperparameters.training.use_core'] = 1
                filter_dict_robust_optim_50['lambda_core'] = 0
                filter_dict_robust_optim_50['worstofk'] = k
                results_robust_50AT = extract_data(results_dict, x_par, y_par,
                    filter_by_dict=filter_dict_robust_optim_50)

                p_50, = plt.plot(results_core[0], 
                    [results_robust_50AT[1]]*len(xvals), 
                    label = '{} with k = {}'.format(y_legend_list[1], k))
                list_legend_handles.append(p_50)
                p_100, = plt.plot(results_core[0], 
                    [results_robust_100AT[1]]*len(xvals),
                    label = '{} with k = {}'.format(y_legend_list[2], k))
                list_legend_handles.append(p_100)
        
        if y_par == 'test_grid_accuracy':
            legend_position = 'upper left'
            bbox_to_anchor=(0., 1.6)
        elif y_par == 'test_nat_accuracy':
            legend_position = 'upper left'
            bbox_to_anchor=(0., 1.4)
        if add_baseline:
            p_base, = plt.plot(xvals, len(xvals)*[baseline_y], 
                fmt_dict['Standard'], label = 'Standard',
                linewidth=linewidth, markersize=markersize)
            list_legend_handles.append(p_base)
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h 
                       for h in handles]
            
            ax.legend(handles=handles, labels=labels, loc=legend_position, ncol = 1,
                bbox_to_anchor=bbox_to_anchor)
        else:
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h 
                for h in list_legend_handles]
            plt.legend(handles=handles, 
                loc=legend_position, bbox_to_anchor=(1, 0.5))        	

        if axis is None:
            axis = get_axis_range(results_core, lowery=0, uppery=100)	
        
    else:
        raise NotImplementedError

    if y_par == 'test_nat_accuracy':
        if dataset == "SVHN":
            axis[2] = 90
            axis[3] = 100
        elif dataset == "CIFAR-10":
            axis[2] = 80
            axis[3] = 100
        elif dataset == "CIFAR-100":
            axis[2] = 50
            axis[3] = 85

    if y_par == 'test_grid_accuracy':
        if dataset == "CIFAR-100":
            axis[2] = 0
            axis[3] = 65
        elif dataset == "CIFAR-10":
            axis[2] = 0
            axis[3] = 90

    plt.axis(axis)
    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    #plt.xticks(xvals)
    plt.grid(grid, linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_acc_vs_runtime(results_dict, 
                        x_par, y_par,
                        worstofk=[1, 5, 10, 20],
                        filter_dict_core = {}, 
                        filter_dict_robust_optim_50 = {}, 
                        filter_dict_robust_optim_100 = {}, 
                        select_best_lambda=False,
                        select_best_lambda_metric='test_grid_accuracy',
                        title = "", 
                        xlabel = "", 
                        ylabel = "",
                        y_legend_list=['CoRe', 'Worst-of-k (50)', 'Worst-of-k (100)'], 
                        fmt_dict={'core': 'r.-', 'rob50': 'g,-.', 'rob100': 'bo--'},
                        marker_dict={1: 'o', 5: '^', 10: '*', 20: 'v'},
                        color_dict={'core': 'r', 'rob50': 'g', 'rob100': 'b'},
                        dataset=None,
                        linewidth=1,
                        markersize=4,
                        axis=None, 
                        grid=True,
                        figsize=(8,6),
                        save_to_file=False, 
                        save_path='.',  
                        save_name='accuracies.pdf'):

    list_legend_handles = []    
    y_legend_list_local = []
    plt.figure(2, figsize=figsize)
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
    
    core_x_list = []
    core_y_list = []
    y_legend_list_core = []

    rob50_x_list = []
    rob50_y_list = []
    y_legend_list_rob50 = []

    rob100_x_list = []
    rob100_y_list = []
    y_legend_list_rob100 = []

    for _, k in enumerate(worstofk):
        # core results
        filter_dict_core['worstofk'] = k
        filter_dict_core['hyperparameters.training.use_core'] = 1
        if select_best_lambda:
            results_core, sel_lam = extract_data_select_best(results_dict, 
                'worstofk', y_par, 'lambda_core', select_best_lambda_metric,
                filter_by_dict=filter_dict_core)
            print('Selected lambdas')
            print(sel_lam)
            filter_dict_core['lambda_core'] = sel_lam[k]
            
        results_core = extract_data(results_dict, x_par, y_par, 
            filter_by_dict=filter_dict_core, average=False)
        runtime = np.mean(results_core[0])
        acc = np.mean(results_core[1])
        core_x_list.append(runtime)
        core_y_list.append(acc)
        
        # robust optim 100 AT results
        filter_dict_robust_optim_100['worstofk'] = k
        filter_dict_robust_optim_100['hyperparameters.training.use_core'] = 0
        results_robust_100AT = extract_data(results_dict, x_par, y_par,
            filter_by_dict=filter_dict_robust_optim_100)
        
        runtime = np.mean(results_robust_100AT[0])
        acc = np.mean(results_robust_100AT[1])
        rob100_x_list.append(runtime)
        rob100_y_list.append(acc)

        # robust optim 50 AT results
        filter_dict_robust_optim_50['worstofk'] = k
        filter_dict_robust_optim_50['hyperparameters.training.use_core'] = 1
        filter_dict_robust_optim_50['lambda_core'] = 0
        results_robust_50AT = extract_data(results_dict, x_par, y_par,
            filter_by_dict=filter_dict_robust_optim_50)
        runtime = np.mean(results_robust_50AT[0])
        acc = np.mean(results_robust_50AT[1])
        rob50_x_list.append(runtime)
        rob50_y_list.append(acc)
    
    p, = plt.plot(core_x_list, core_y_list, fmt_dict['core'], 
            markersize=markersize)
    list_legend_handles.append(p)
    
    p, = plt.plot(rob50_x_list, rob50_y_list, fmt_dict['rob50'], 
            markersize=markersize)
    list_legend_handles.append(p)

    p, = plt.plot(rob100_x_list, rob100_y_list, fmt_dict['rob100'], 
            markersize=markersize)
    list_legend_handles.append(p)

    # xvals = rob100_x_list
    # xvals[0] = 1
    # xvals[-1] = 12
    # plt.plot(xvals, [max(rob100_y_list)]*len(rob100_x_list),
    #     color='black', linestyle=":")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)			
   
    legend_position = 'lower right'
    # bbox_to_anchor=(1.9, 1.)
    
    plt.legend(list_legend_handles, y_legend_list,
        loc=legend_position) #, bbox_to_anchor=bbox_to_anchor)
    
    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    
    ax.set_yticks([max(rob100_y_list)], minor=True)
    plt.grid(grid, linestyle=':')
    ax.yaxis.grid(True, color = 'black', which='minor', linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_acc_vs_runtime2(results_dict, 
                        x_par, y_par,
                        worstofk=[1, 5, 10, 20],
                        filter_dict_core = {}, 
                        filter_dict_robust_optim_50 = {}, 
                        filter_dict_robust_optim_100 = {}, 
                        select_best_lambda=False,
                        select_best_lambda_metric='test_grid_accuracy',
                        title = "", 
                        xlabel = "", 
                        ylabel = "",
                        y_legend_list=['CoRe', 'Worst-of-k (50)', 'Worst-of-k (100)'], 
                        fmt_dict={'CoRe': 'r.-', 'Worst-of-k (50)': 'g,-.', 'Worst-of-k (100)': 'bo--'},
                        marker_dict={1: 'o', 5: '^', 10: '*', 20: 'v'},
                        color_dict={'core': 'r', 'rob50': 'g', 'rob100': 'b'},
                        dataset=None,
                        linewidth=1,
                        markersize=4,
                        axis=None, 
                        grid=True,
                        figsize=(8,6),
                        save_to_file=False, 
                        save_path='.',  
                        save_name='accuracies.pdf'):

    list_legend_handles = []    
    y_legend_list_local = []
    plt.figure(2, figsize=figsize)
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
    
    for _, k in enumerate(worstofk):
        # core results
        filter_dict_core['worstofk'] = k
        filter_dict_core['hyperparameters.training.use_core'] = 1
        if select_best_lambda:
            results_core, sel_lam = extract_data_select_best(results_dict, 
                'worstofk', y_par, 'lambda_core', select_best_lambda_metric,
                filter_by_dict=filter_dict_core)
            print('Selected lambdas')
            print(sel_lam)
            filter_dict_core['lambda_core'] = sel_lam[k]
            
        results_core = extract_data(results_dict, x_par, y_par, 
            filter_by_dict=filter_dict_core, average=False)
        runtime = np.mean(results_core[0])
        acc = np.mean(results_core[1])
        y_legend_list_local.append("{}{}".format(y_legend_list[0], k))
        p, = plt.plot([runtime], [acc], color=color_dict['core'], 
            marker=marker_dict[k])
        list_legend_handles.append(p)

        
        # robust optim 100 AT results
        filter_dict_robust_optim_100['worstofk'] = k
        filter_dict_robust_optim_100['hyperparameters.training.use_core'] = 0
        results_robust_100AT = extract_data(results_dict, x_par, y_par,
            filter_by_dict=filter_dict_robust_optim_100)
        
        runtime = np.mean(results_robust_100AT[0])
        acc = np.mean(results_robust_100AT[1])
        p, = plt.plot([runtime], [acc], color=color_dict['rob100'], 
            marker=marker_dict[k])
        list_legend_handles.append(p)
        y_legend_list_local.append("{}{}".format(y_legend_list[2], k))

        # robust optim 50 AT results
        filter_dict_robust_optim_50['worstofk'] = k
        filter_dict_robust_optim_50['hyperparameters.training.use_core'] = 1
        filter_dict_robust_optim_50['lambda_core'] = 0
        results_robust_50AT = extract_data(results_dict, x_par, y_par,
            filter_by_dict=filter_dict_robust_optim_50)
        runtime = np.mean(results_robust_50AT[0])
        acc = np.mean(results_robust_50AT[1])
        p, = plt.plot([runtime], [acc], color=color_dict['rob50'], 
            marker=marker_dict[k])
        list_legend_handles.append(p)
        y_legend_list_local.append("{}{}".format(y_legend_list[1], k)) 
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)			
   
    legend_position = 'upper right'
    bbox_to_anchor=(1.9, 1.)
    
    plt.legend(list_legend_handles, y_legend_list_local,
        loc=legend_position, bbox_to_anchor=bbox_to_anchor)
    
    ax = plt.gca()
    adjust_spines(ax, ['left', 'bottom'])
    plt.grid(grid, linestyle=':')
    if save_to_file:
        plt.savefig(os.path.join(save_path, save_name), 
            pad_inches=.1, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
