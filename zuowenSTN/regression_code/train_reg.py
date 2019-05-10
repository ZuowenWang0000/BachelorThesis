"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import shutil
import click
import sys
import copy
import time
import getpass
from timeit import default_timer as timer
import ipdb

import numpy as np
import tensorflow as tf

import cifar10_input_reg as cifar10_input
import cifar100_input
import svhn_input
from eval import evaluate
import experiment_repo as exprepo
import resnet_reg
import vgg
from spatial_attack import SpatialAttack
import utilities
import pickle


def train(config='configs/cifar10_regression.json',
          save_root_path='/cluster/work/math/fanyang-broglil/CoreRepo',
          experiment_json_fname='experiments.json',
          local_json_dir_name='local_json_files',
          worstofk=None,
          attack_style=None,
          attack_limits=None,
          fo_epsilon=None,
          fo_step_size=None,
          fo_num_steps=None,
          lambda_core=None,
          num_ids = None,
          group_size=None,
          use_core=None,
          seed=None,
          save_in_local_json=True,
          this_repo=None):

    # reset default graph (needed for running locally with run_jobs_ray.py)
    tf.reset_default_graph()

    # get configs
    config_dict = utilities.get_config(config)
    config_dict_copy = copy.deepcopy(config_dict)
    config = utilities.config_to_namedtuple(config_dict)

    # seeding randomness
    if seed == None:
        seed = config.training.tf_random_seed
    else:
        config_dict_copy['training']['tf_random_seed'] = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Setting up training parameters
    max_num_training_steps = config.training.max_num_training_steps
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum

    if group_size == None:
        group_size = config.training.group_size
    else:
        config_dict_copy['training']['group_size'] = int(group_size)
    if num_ids == None:
        num_ids = config.training.num_ids
    else:
        config_dict_copy['training']['num_ids'] = int(num_ids)
    if lambda_core == None:
        lambda_core = config.training.lambda_
    else:
        config_dict_copy['training']['lambda_'] = float(lambda_core)
    if use_core == None:
        use_core = config.training.use_core
    else:
        config_dict_copy['training']['use_core'] = use_core

    batch_size = config.training.batch_size
    # number of groups with group size > 1
    num_grouped_ids = batch_size - num_ids
    # number of unique ids needs to be larger than half the desired batch size
    # so that full batch can be filled up
    assert num_ids >= batch_size/2
    # currently, code is designed for groups of size 2
    assert config.training.group_size == 2

    adversarial_training = config.training.adversarial_training
    eval_during_training = config.training.eval_during_training
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    # Setting up output parameters
    num_output_steps = config.training.num_output_steps
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps
    num_easyeval_steps = config.training.num_easyeval_steps

    # Setting up the data and the model
    data_path = config.data.data_path

    if config.data.dataset_name == "cifar-10":
        raw_iterator = cifar10_input.CIFAR10Data(data_path)
    elif config.data.dataset_name == "cifar-100":
        raw_iterator = cifar100_input.CIFAR100Data(data_path)
    elif config.data.dataset_name == "svhn":
        raw_iterator = svhn_input.SVHNData(data_path)
    else:
        raise ValueError("Unknown dataset name.")

    global_step = tf.train.get_or_create_global_step()

    model_family = config.model.model_family
    if model_family == "resnet":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = resnet_reg.Model(config.model, num_ids, diffable,
            config.training.adversarial_ce)
    elif model_family == "vgg":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        if config.training.adversarial_ce:
            raise NotImplementedError
        model = vgg.Model(config.model, num_ids, diffable)

    # uncomment to get a list of trainable variables
    # model_vars = tf.trainable_variables()

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)

    if use_core and lambda_core > 0:
        print("WARNING: in regression task, should not enter this section!\n")
        total_loss = (model.reg_loss + weight_decay * model.weight_decay_loss +
                      lambda_core * model.core_loss2)
    else:
        total_loss = model.reg_loss + weight_decay * model.weight_decay_loss

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_step = optimizer.minimize(total_loss, global_step=global_step)

    # Set up adversary
    if worstofk == None:
        worstofk = config.attack.random_tries
    else:
        config_dict_copy['attack']['random_tries'] = worstofk
    if fo_epsilon == None:
        fo_epsilon = config.attack.epsilon
    else:
        config_dict_copy['attack']['epsilon'] = fo_epsilon
    if fo_step_size == None:
        fo_step_size = config.attack.step_size
    else:
        config_dict_copy['attack']['step_size'] = fo_step_size
    if fo_num_steps == None:
        fo_num_steps = config.attack.num_steps
    else:
        config_dict_copy['attack']['num_steps'] = fo_num_steps
    # @ Luzius: incorporate being able to choose multiple transformations
    if attack_style == None:
        attack_style = 'rotate'

    simple_train = config.attack.simple_train

    if simple_train == False:

        # Training attack == denfense
        # L-inf attack if use_spatial is False and use_linf is True
        # spatial attack if use_spatial is True and use_linf is False
        # spatial random attack if spatial_method is 'random'
        # spatial PGD attack if spatial_method is 'fo'
        attack = SpatialAttack(model, config.attack, config.attack.spatial_method,
                               worstofk, attack_limits, fo_epsilon,
                               fo_step_size, fo_num_steps)
        # Different eval attacks
        # Random attack
        # L-inf attack if use_spatial is False and use_linf is True
        # random (worst-of-1) spatial attack if use_spatial is True
        # and use_linf is False
        attack_eval_random = SpatialAttack(model, config.attack, 'random', 1,
                                           attack_limits, fo_epsilon,
                                           fo_step_size, fo_num_steps)
        # First order attack
        # L-inf attack if use_spatial is False and use_linf is True
        # first-order spatial attack if use_spatial is True and use_linf is False
        attack_eval_fo = SpatialAttack(model, config.attack, 'fo', 1,
                                       attack_limits, fo_epsilon,
                                       fo_step_size, fo_num_steps)

        # Grid attack
        # spatial attack if use_spatial is True and use_linf is False
        # not executed for L-inf attacks
        attack_eval_grid = SpatialAttack(model, config.attack, 'grid', None,
                                         attack_limits)
    else:
        attack = SpatialAttack(model, config.attack, config.attack.spatial_method,
                               worstofk, attack_limits, fo_epsilon,
                               fo_step_size, fo_num_steps)

    # TODO(christina): add L-inf attack with random restarts

    # ------------------START EXPERIMENT -------------------------
    # Initialize the Repo
    print("==> Creating repo..")
    # Create repo object if it wasn't passed, comment out if repo has issues
    if this_repo == None:
        this_repo = exprepo.ExperimentRepo(
            save_in_local_json=save_in_local_json,
            json_filename=experiment_json_fname,
            local_dir_name=local_json_dir_name,
            root_dir=save_root_path)

    # Create new experiment
    if this_repo != None:
        exp_id = this_repo.create_new_experiment(config.data.dataset_name,
                                                 model_family,
                                                 worstofk,
                                                 attack_style,
                                                 attack_limits,
                                                 lambda_core,
                                                 num_grouped_ids,
                                                 group_size,
                                                 config_dict_copy)

    # Setting up the Tensorboard and checkpoint outputs
    model_dir = '%s/logdir/%s' % (save_root_path, exp_id)

    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=3)

    tf.summary.scalar('regression loss function value', model.reg_loss, collections= ['err'])
    tf.summary.scalar('avg_abs_err_x', model.avg_abs_err_transX, collections=['err'])
    tf.summary.scalar('avg_abs_err_y', model.avg_abs_err_transY, collections=['err'])
    tf.summary.scalar('avg_abs_err_rot', model.avg_abs_err_rot, collections=['err'])
    tf.summary.scalar('avg_rel_err_x', model.avg_rel_err_transX, collections=['err'])
    tf.summary.scalar('avg_rel_err_y', model.avg_rel_err_transY, collections=['err'])
    tf.summary.scalar('avg_rel_err_rot', model.avg_rel_err_rot, collections=['err'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['err'])
    tf.summary.image('before_reflect_padding', model.before_reflect_x, collections=['err'])
    tf.summary.image('after_reflect_padding', model.reflect_x, collections=['err'])

    err_summaries = tf.summary.merge_all('err')

    tf.summary.scalar('full_batch_avg_abs_err_x', model.avg_abs_err_transX, collections=['eval'])
    tf.summary.scalar('full_batch_avg_abs_err_y', model.avg_abs_err_transY, collections=['eval'])
    tf.summary.scalar('full_batch_avg_abs_err_rot', model.avg_abs_err_rot, collections=['eval'])
    tf.summary.scalar('full_batch_avg_rel_err_x', model.avg_rel_err_transX, collections=['eval'])
    tf.summary.scalar('full_batch_avg_rel_err_y', model.avg_rel_err_transY, collections=['eval'])
    tf.summary.scalar('full_batch_avg_rel_err_rot', model.avg_rel_err_rot, collections=['eval'])
    tf.summary.scalar('full_batch_avg_worst', model.avg_worst_err, collections=['eval'])

    # tf.summary.scalar('learning_rate', learning_rate, collections=['eval'])
    eval_summaries = tf.summary.merge_all('eval')


    # data augmentation used if config.training.data_augmentation_core is True
    x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                        x_input_placeholder)

    with tf.Session() as sess:
        # initialize standard data augmentation
        if config.training.data_augmentation:
            if config.data.dataset_name == "cifar-10":
                data_iterator = cifar10_input.AugmentedCIFAR10Data(raw_iterator, sess)
            elif config.data.dataset_name == "cifar-100":
                data_iterator = cifar100_input.AugmentedCIFAR100Data(raw_iterator, sess)
            elif config.data.dataset_name == "svhn":
                data_iterator = svhn_input.AugmentedSVHNData(raw_iterator, sess)
            else:
                raise ValueError("Unknown dataset name.")
        else:
            data_iterator = raw_iterator

        if simple_train:
            # attack = SpatialAttack(model, config.attack, config.attack.spatial_method,
            #                        worstofk, attack_limits, fo_epsilon,
            #                        fo_step_size, fo_num_steps)
            # attack.simple_train_perturb should return a list of parameters (len(x_batch),3)
            x_eval_batch = data_iterator.eval_data.xs
            x_batch_eval = x_eval_batch
            # the evaluation batch labels are 3 dim transformations now
            y_batch_eval = data_iterator.eval_data.ys
            # we pass the label values to the model. as the transformation
            trans_eval = y_batch_eval
            eval_dict = {model.x_input: x_batch_eval,
                         model.y_input: y_batch_eval,
                         # group is not used in simple train
                         model.group:  np.arange(0, batch_size, 1, dtype="int32"),
                         model.transform: trans_eval,
                         model.is_training: False}

        else:
            eval_dict = {model.x_input: data_iterator.eval_data.xs,
                         model.y_input: data_iterator.eval_data.ys,
                         model.group:  np.arange(0, batch_size, 1, dtype="int32"),
                         model.transform: np.zeros([data_iterator.eval_data.n, 3]),
                         model.is_training: False}

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
        # if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        # eval_summary_writer = tf.summary.FileWriter(eval_dir)

        sess.run(tf.global_variables_initializer())
        training_time = 0.0
        run_time_without_eval = 0.0
        run_time_adv_ex_creation = 0.0
        run_time_train_step = 0.0
        ####################################
        # Main training loop
        ####################################
        start_time = time.time()
        no_epochs_done = 0 # the same as epoch_count, need to merge
        start_epoch = timer()
        it_count = 0
        epoch_count = 0
        acc_sum = 0

        printFlag = 0

        for ii in range(max_num_training_steps+1):
            # original batch
            x_batch, y_batch, epoch_done = data_iterator.train_data.get_next_batch(
                num_ids, multiple_passes=True)
            no_epochs_done += epoch_done
            # noop trans
            noop_trans = np.zeros([len(x_batch), 3])
            # id_batch starts with IDs of original examples
            id_batch = np.arange(0, num_ids, 1, dtype="int32")

            if use_core:
                print("*********Warning: should not be using core in train_reg.py!*********\n")
            else:
                if adversarial_training:
                    start = timer()

                    if simple_train:

                        # only generating 1 tilted image per original image, which is equivalent to wo-1
                        x_batch_inp = x_batch
                        # using the labels as the transformation, hope the model will learn it
                        trans_inp = y_batch

                    else:
                        print("shouldn't be entering here in regression task!\n")
                        quit()
                        x_batch_inp, trans_inp = attack.perturb(x_batch, y_batch,
                                                            sess)
                    end = timer()
                    training_time += end - start
                    run_time_without_eval += end - start
                    run_time_adv_ex_creation += end - start
                else:
                    x_batch_inp, trans_inp = x_batch, noop_trans

                # if simple_train:
                #     y_batch_inp = y_batch
                #     y_batch_adv = transform_parameters
                #     trans_adv = transform_parameters
                #     x_batch_adv = x_batch_inp
                #     id_batch_inp = id_batch
                #     id_batch_adv = id_batch
                # for adversarial training and plain training, the following
                # variables coincide
                # else:
                y_batch_inp = y_batch
                y_batch_adv = y_batch
                trans_adv = trans_inp
                x_batch_adv = x_batch_inp
                id_batch_inp = id_batch
                id_batch_adv = id_batch

            # feed_dict for training step
            inp_dict = {model.x_input: x_batch_inp,
                        model.y_input: y_batch_inp,
                        model.group: id_batch_inp,
                        model.transform: trans_inp,
                        model.is_training: False}

            # separate natural and adversarially transformed examples for eval
            # nat_dict = {model.x_input: x_batch,
            #             model.y_input: y_batch,
            #             model.group: id_batch,
            #             model.transform: noop_trans,
            #             model.is_training: False}
            #
            # adv_dict = {model.x_input: x_batch_adv,
            #             model.y_input: y_batch_adv,
            #             model.group: id_batch_adv,
            #             model.transform: trans_adv,
            #             model.is_training: False}

            loss = sess.run(model.reg_loss, feed_dict=inp_dict)
            if ii % num_easyeval_steps == 0 or ii == max_num_training_steps:
                print("\nin resnet_reg")
                print("y_input")
                y_input = sess.run(model.y_input, feed_dict=inp_dict)
                print(y_input[0:5])
                print("\nprediction")
                prediction = sess.run(model.prediction, feed_dict=inp_dict)
                print(prediction[0:5])
                print(' Easy Evalutaion Step training time error  Training Set   ')
                print('avg_abs_err_transX : {}'.format(sess.run(model.avg_abs_err_transX, feed_dict=inp_dict)))
                print('avg_abs_err_transY : {}'.format(sess.run(model.avg_abs_err_transY, feed_dict=inp_dict)))
                print('avg_abs_err_rot : {}'.format(sess.run(model.avg_abs_err_rot, feed_dict=inp_dict)))
                print('avg_rel_err_transX : {}'.format(sess.run(model.avg_rel_err_transX, feed_dict=inp_dict)))
                print('avg_rel_err_transY : {}'.format(sess.run(model.avg_rel_err_transY, feed_dict=inp_dict)))
                print('avg_rel_err_rot : {}'.format(sess.run(model.avg_rel_err_rot, feed_dict=inp_dict)))

                print(' Easy Evalutaion Step training time error  Evaluation Set   ')
                print('avg_abs_err_transX : {}'.format(sess.run(model.avg_abs_err_transX, feed_dict=eval_dict)))
                print('avg_abs_err_transY : {}'.format(sess.run(model.avg_abs_err_transY, feed_dict=eval_dict)))
                print('avg_abs_err_rot : {}'.format(sess.run(model.avg_abs_err_rot, feed_dict=eval_dict)))
                print('avg_rel_err_transX : {}'.format(sess.run(model.avg_rel_err_transX, feed_dict=eval_dict)))
                print('avg_rel_err_transY : {}'.format(sess.run(model.avg_rel_err_transY, feed_dict=eval_dict)))
                print('avg_rel_err_rot : {}'.format(sess.run(model.avg_rel_err_rot, feed_dict=eval_dict)))
            # Output to stdout
            if epoch_done:
                epoch_time = timer() - start_epoch

                # ToDo: Log this to file as well

                # Training accuracy over epoch
                print('Epoch {}:    ({})'.format(epoch_count, datetime.now()))
                print('    training loss {:.4}'.format(loss))
                print('    {:.4} seconds per epoch'.format(epoch_time))

                print('    training time error  Training Set   ')
                print('avg_abs_err_transX : {}'.format(sess.run(model.avg_abs_err_transX, feed_dict=inp_dict)))
                print('avg_abs_err_transY : {}'.format(sess.run(model.avg_abs_err_transY, feed_dict=inp_dict)))
                print('avg_abs_err_rot : {}'.format(sess.run(model.avg_abs_err_rot, feed_dict=inp_dict)))
                print('avg_rel_err_transX : {}'.format(sess.run(model.avg_rel_err_transX, feed_dict=inp_dict)))
                print('avg_rel_err_transY : {}'.format(sess.run(model.avg_rel_err_transY, feed_dict=inp_dict)))
                print('avg_rel_err_rot : {}'.format(sess.run(model.avg_rel_err_rot, feed_dict=inp_dict)))

                # if ii % config.eval.full_batch_eval_steps == 0:
                #     print_eval_fullbatch(ii, sess, y_batch_eval, batch_size, trans_eval, model, attack, model_dir, global_step, summary_writer)

                epoch_done = 0
                epoch_count += 1
                start_epoch = timer()
                it_count = 1
            else:
                it_count += 1
                # acc_sum += nat_acc_tr


            # Output to stdout
            if ii % num_output_steps == 0:
                if ii != 0:
                     training_time = 0.0

            # Tensorboard summaries and heavy checkpoints
            if ii % num_summary_steps == 0:
                summary = sess.run(err_summaries, feed_dict=inp_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))


            # Write a checkpoint and eval if it's time
            if ii % num_checkpoint_steps == 0 or ii == max_num_training_steps:
                # Save checkpoint data (weights)
                saver.save(sess,
                           os.path.join(model_dir, 'checkpoint'),
                           global_step=global_step)

                full_eval_dict = fetch_full_eval_dict(model, batch_size)

                print_eval_fullbatch(ii,sess, model, model_dir, full_eval_dict)
                summary_eval = sess.run(eval_summaries, feed_dict=full_eval_dict)
                summary_writer.add_summary(summary_eval, global_step.eval(sess))

                # Evaluation on full evaluation batch
                # if ((eval_during_training and ii % num_eval_steps == 0
                #     and ii > 0 and config.attack.use_spatial) or
                #     (eval_during_training and ii == max_num_training_steps and
                #     config.attack.use_spatial)):
                #     print_eval_fullbatch(ii,sess, y_batch_eval, batch_size, trans_eval, model, attack, model_dir, global_step, summary_writer)

            # Actual training step
            start = timer()
            inp_dict[model.is_training] = True
            sess.run(train_step, feed_dict=inp_dict)
            end = timer()
            training_time += end - start
            run_time_without_eval += end - start
            run_time_train_step += end - start

        runtime = time.time() - start_time

        # this_repo.mark_experiment_as_completed(
        #     runtime=runtime, runtime_wo_eval=run_time_without_eval,
        #     runtime_train_step=run_time_train_step,
        #     runtime_adv_ex_creation=run_time_adv_ex_creation)

    return 0

def fetch_full_eval_dict(model, batch_size):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    test_batch = unpickle('./datasets/cifar10/test_batch')
    features = test_batch[b'data']
    features = features.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    label = test_batch[b'labels']

    full_eval_dict = {model.x_input: features,
                 model.y_input: label,
                 # group is not used in simple train
                 model.group: np.arange(0, batch_size, 1, dtype="int32"),
                 model.transform: label,
                 model.is_training: False}

    return full_eval_dict

def print_eval_fullbatch(ii, sess, model,  model_dir, full_eval_dict):
    # full_eval_trans = attack.eval_perturb(features)
    print('########## Full Eval.Batch Evaluation at training step: {} ###########'.format(ii))

    eval_avg_abs_err_transX = sess.run(model.avg_abs_err_transX, feed_dict=full_eval_dict)
    eval_avg_abs_err_transY = sess.run(model.avg_abs_err_transY, feed_dict=full_eval_dict)
    eval_avg_abs_err_rot = sess.run(model.avg_abs_err_rot, feed_dict=full_eval_dict)
    eval_avg_rel_err_transX = sess.run(model.avg_rel_err_transX, feed_dict=full_eval_dict)
    eval_avg_rel_err_transY = sess.run(model.avg_rel_err_transY, feed_dict=full_eval_dict)
    eval_avg_rel_err_rot = sess.run(model.avg_rel_err_rot, feed_dict=full_eval_dict)
    eval_avg_rel_err_worst = sess.run(model.avg_worst_err, feed_dict=full_eval_dict)

    print('full_batch_avg_abs_err_transX : {}'.format(eval_avg_abs_err_transX))
    print('full_batch_avg_abs_err_transY : {}'.format(eval_avg_abs_err_transY))
    print('full_batch_avg_abs_err_rot : {}'.format(eval_avg_abs_err_rot))
    print('full_batch_avg_rel_err_transX : {}'.format(eval_avg_rel_err_transX))
    print('full_batch_avg_rel_err_transY : {}'.format(eval_avg_rel_err_transY))
    print('full_batch_avg_rel_err_rot : {}'.format(eval_avg_rel_err_rot))
    print('full_batch_avg_worst : {}'.format(eval_avg_rel_err_worst))

    err_x_rel = sess.run(model.err_x_rel, feed_dict=full_eval_dict)
    err_y_rel = sess.run(model.err_y_rel, feed_dict=full_eval_dict)
    err_rot_rel = sess.run(model.err_rot_rel, feed_dict=full_eval_dict)
    err_worst_rel = sess.run(model.err_worst_rel, feed_dict=full_eval_dict)

    print("histogram err_x_rel")
    plot_hist(err_x_rel, 20)
    print("histogram err_y_rel")
    plot_hist(err_y_rel, 20)
    print("histogram err_rot_rel")
    plot_hist(err_rot_rel, 20)

    print("histogram err_x_rel")
    plot_hist(err_x_rel, 10)
    print("histogram err_y_rel")
    plot_hist(err_y_rel, 10)
    print("histogram err_rot_rel")
    plot_hist(err_rot_rel, 10)

    total_rel_err = np.vstack((err_x_rel, err_y_rel, err_rot_rel, err_worst_rel))
    total_rel_err = np.transpose(total_rel_err)
    file_name = "csv_rel_err_" + str(ii) + ".csv"
    saveCSVpath = os.path.join(model_dir, file_name)
    np.savetxt(saveCSVpath, total_rel_err, delimiter=",")
    eval_dir = os.path.join(model_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)


def plot_hist(err_rel, num_bins = 10):
    min = np.min(err_rel)
    max = np.max(err_rel)
    print("max rel err: {}".format(max))
    print("min rel err: {}".format(min))
    counts, bins = np.histogram(err_rel, bins=num_bins, range=(min, max))
    print(bins)
    print(counts)


def load_cfar10_test_batch(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path, mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

@click.command()
@click.option('--config', default='configs/cifar10_regression.json', type=str)
@click.option('--save-root-path',
              default='/cluster/work/math/fanyang-broglil/CoreRepo',
              help='path to project root dir')
@click.option('--experiment_json_fname',
              default='experiments.json',
              help='filename for json saving experimental results')
@click.option('--local_json_dir_name',
              default='local_json_files',
              help='foldername for local json files')
@click.option('--save_in_local_json', default=1, type=int)
@click.option('--worstofk', default=None, type=int)
@click.option('--attack-style', default=None, type=str,
              help='Size multipler for original CIFAR dataset')
# ToDo: should be an Array, currently unused
@click.option('--attack-limits', default=None)
@click.option('--lambda-core', default=None, type=float)
@click.option('--fo_epsilon', default=None, type=float)
@click.option('--fo_step_size', default=None, type=float)
@click.option('--fo_num_steps', default=None, type=int)
@click.option('--num-ids', default=None, type=int)
@click.option('--group-size', default=None, type=int)
@click.option('--use_core', default=None, type=bool)
@click.option('--seed', default=None, type=int)
# @click.option('--seed', default=None, type=int)
# @click.option('--simple_train', default=True, type=bool)

def train_cli(config, save_root_path, experiment_json_fname, local_json_dir_name,
              worstofk, attack_style, attack_limits, fo_epsilon,
              fo_step_size, fo_num_steps,
              lambda_core, num_ids, group_size, use_core, seed,
              save_in_local_json):
    '''Train a ResNet on Cifar10.'''
    train(config, save_root_path, experiment_json_fname, local_json_dir_name,
          worstofk, attack_style, attack_limits, fo_epsilon,
          fo_step_size, fo_num_steps,
          lambda_core, num_ids, group_size, use_core, seed, save_in_local_json)

if __name__ == '__main__':
    train_cli()
