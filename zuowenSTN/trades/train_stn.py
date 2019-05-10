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

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim

import cifar10_input
import cifar100_input
from eval import evaluate
import resnet
import stn_resnet
import stn_resnet_conv
from spatial_attack import SpatialAttack
import utilities

import experiment_repo as exprepo

def train(config='configs/cifar10_config_stn.json',
          save_root_path='/cluster/work/math/fanyang-broglil/CoreRepo',
          worstofk=None,
          attack_style=None,
          attack_limits=None,
          lambda_core=None,
          num_grouped_ids=None,
          num_ids = None,
          group_size=None,
          use_core=None,
          seed=None,
          this_repo=None):

    config_dict = utilities.get_config(config)
    config_dict_copy = copy.deepcopy(config_dict)
    # model_dir = config_dict['model']['output_dir']
    # if not os.path.exists(model_dir):
    #   os.makedirs(model_dir)

    # # keep the configuration file with the model for reproducibility
    # with open(os.path.join(model_dir, 'config.json'), 'w') as f:
    #     json.dump(config_dict, f, sort_keys=True, indent=4)

    config = utilities.config_to_namedtuple(config_dict)

    # seeding randomness
    if seed == None:
        seed = config.training.tf_random_seed
    else:
        config_dict_copy['training']['tf_random_seed'] = seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Setting up training parameters
    max_num_epochs = config.training.max_num_epochs
    step_size_schedule = config.training.step_size_schedule
    weight_decay = config.training.weight_decay
    momentum = config.training.momentum
    num_ids = config.training.num_ids # number of IDs per minibatch

    if group_size == None:
        group_size = config.training.group_size
    else:
        config_dict_copy['training']['group_size'] = group_size
    if num_grouped_ids == None:
        num_grouped_ids = config.training.num_grouped_ids
    else:
        config_dict_copy['training']['num_grouped_ids'] = num_grouped_ids
    if num_ids == None:
        num_ids = config.training.num_ids
    else:
        config_dict_copy['training']['num_ids'] = num_ids
    if lambda_core == None:
        lambda_core = config.training.lambda_
    else:
        config_dict_copy['training']['lambda_'] = lambda_core
    if use_core == None:
        use_core = config.training.use_core
    else:
        config_dict_copy['training']['use_core'] = use_core

    adversarial_training = config.training.adversarial_training
    eval_during_training = config.training.eval_during_training
    if eval_during_training:
        num_eval_steps = config.training.num_eval_steps

    # Setting up output parameters
    num_summary_steps = config.training.num_summary_steps
    num_checkpoint_steps = config.training.num_checkpoint_steps
    num_easyeval_steps = config.training.num_easyeval_steps

    # mini batch size per iteration
    # ToDo: need to make this support variable number of num_grouped_ids
    batch_size = num_ids + num_grouped_ids

    # Setting up model and loss
    model_family = config.model.model_family
    with_transformer = config.model.transformer
    translation_model =config.model.translation_model
    if model_family == "resnet":
        if with_transformer == True:
            if translation_model == "fc":
                model = stn_resnet.Model(config.model)
                print("Using stn_resnet")
            if translation_model == "conv":
                model = stn_resnet_conv.Model(config.model)
                print("Using stn_resnet_conv")
        else:
            model = resnet.Model(config.model)
    else:
        print("Model family does not exist")
        exit()
    if use_core:
      total_loss = model.mean_xent + weight_decay * model.weight_decay_loss + lambda_core * model.core_loss2
    else:
      total_loss = model.mean_xent + weight_decay * model.weight_decay_loss


     # Setting up the data and the model
    data_path = config.data.data_path
    
    if config.data.dataset_name == "cifar-10":
        raw_cifar = cifar10_input.CIFAR10Data(data_path)
    elif config.data.dataset_name == "cifar-100":
        raw_cifar = cifar100_input.CIFAR100Data(data_path)
    else:
        raise ValueError("Unknown dataset name.")


    # uncomment to get a list of trainable variables
    # model_vars = tf.trainable_variables()
    # slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    # Setting up the optimizer
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
    #                                   name="Adam")
    train_step = optimizer.minimize( total_loss, global_step=global_step)

    # Set up adversary
    if worstofk == None:
        worstofk = config.attack.random_tries
    else:
        config_dict_copy['attack']['random_tries'] = worstofk
    # @ Luzius: incorporate being able to choose multiple transformations
    if attack_style == None:
        attack_style = 'rotate'

    # Training attack
    attack = SpatialAttack(model, config.attack, 'random', worstofk, attack_limits)
    # Different eval attacks
    # Same attack as worstofk
    # @ Luzius: currently the names are not clear/consistent since I wasn't sure if we actually want random or not since you originally had your attack like that but I feel like it should rather be worstofk?
    # attack_eval_adv = SpatialAttack(model, config.attack, 'random', worstofk, attack_limits)
    attack_eval_random = SpatialAttack(model, config.attack, 'random', 1, attack_limits)
    # Grid attack
    attack_eval_grid = SpatialAttack(model, config.attack, 'grid', None, attack_limits)

    # ------------------START EXPERIMENT -------------------------
    # Initialize the Repo
    print("==> Creating repo..")
    # Create repo object if it wasn't passed, comment out if repo has issues
    if this_repo == None:
        this_repo = exprepo.ExperimentRepo(root_dir=save_root_path)

    # Create new experiment
    if this_repo != None:
        exp_id = this_repo.create_new_experiment('cifar-10',
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
    os.makedirs(model_dir, exist_ok=True)
    # We add accuracy and xent twice so we can easily make three types of
    # comparisons in Tensorboard:
    # - train vs eval (for a single run)
    # - train of different runs
    # - eval of different runs

    saver = tf.train.Saver(max_to_keep=3)

    tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
    tf.summary.scalar('accuracy_nat', model.accuracy, collections = ['nat'])
    tf.summary.scalar('xent_nat_train', model.xent / batch_size,
                                                        collections=['nat'])
    tf.summary.scalar('xent_nat', model.xent / batch_size, collections=['nat'])
    tf.summary.image('images_nat_train', model.x_image, collections=['nat'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
    tf.summary.scalar('regression_loss', model.reg_loss, collections=['nat'])
    nat_summaries = tf.summary.merge_all('nat')

    #dataAugmentation
    x_input_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 32, 32, 3])
    flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                            x_input_placeholder)

    tot_samp = raw_cifar.train_data.n
    max_num_iterations = int(np.floor((tot_samp/num_ids)*max_num_epochs))
    print("Total # of samples is: %d; This exp. will run %d iterations" % (tot_samp, max_num_iterations))

    # Compute the (epoch) gaps between summary, worstof1eval, checkpoints should happen
    summary_gap = int(np.floor(max_num_epochs/num_summary_steps))
    easyeval_gap = int(np.floor(max_num_epochs/num_easyeval_steps))
    checkpoint_gap = int(np.floor(max_num_epochs/num_checkpoint_steps))

    with tf.Session() as sess:


        # initialize data augmentation
        if config.training.data_augmentation:
            if config.data.dataset_name == "cifar-10":
                cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess)
            elif config.data.dataset_name == "cifar-100":
                cifar = cifar100_input.AugmentedCIFAR100Data(raw_cifar, sess)
            else:
                raise ValueError("Unknown dataset name.")
        else:
            cifar = raw_cifar

            
        cifar_eval_dict = {model.x_input: cifar.eval_data.xs,
                           model.y_input: cifar.eval_data.ys,
                           model.group:  np.arange(0, batch_size, 1, dtype="int32"),
                           model.transform: np.zeros([cifar.eval_data.n, 3]),
                           model.is_training: False}


        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
        #if eval_during_training:
        eval_dir = os.path.join(model_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        eval_summary_writer = tf.summary.FileWriter(eval_dir)

        sess.run(tf.global_variables_initializer())
        
        training_time = 0.0

        ####################################
        # Main training loop
        ####################################
        # Initialize cache variables
        start_time = time.time()
        start_epoch = timer()
        it_count = 0
        epoch_count = 0
        acc_sum = 0
        it_summary = 0
        it_easyeval = 0
        it_ckpt = 0
        adv_time = 0
        train_time = 0

        for ii in range(max_num_iterations+1):
            x_batch, y_batch, epoch_done = cifar.train_data.get_next_batch(num_ids, multiple_passes=True)


            noop_trans = np.zeros([len(x_batch), 3])
            x_batch_nat = x_batch
            y_batch_nat = y_batch
            id_batch = np.arange(0, num_ids, 1, dtype="int32")
            if use_core:
                # Create rotated examples
                start = timer()
                ids = np.arange(0,num_grouped_ids,1,dtype="int32")

                for i in range(config.training.group_size):
                     
                   if config.training.data_augmentation_core:
                       x_batch_core = sess.run(flipped,feed_dict={x_input_placeholder: x_batch[0:num_grouped_ids,:,:,:]})
                   else:
                       x_batch_core = x_batch[0:num_grouped_ids,:,:,:]

                   x_batch_group, trans_group = attack.perturb(x_batch_core, y_batch[0:num_grouped_ids], sess)

                   #construct new batches including rotated examples
                   x_batch_nat = np.concatenate((x_batch_nat, x_batch_group), axis=0)
                   y_batch_nat = np.concatenate((y_batch_nat, y_batch), axis=0)
                   noop_trans = np.concatenate((noop_trans, trans_group), axis=0)
                   id_batch = np.concatenate((id_batch, ids), axis=0)

                end = timer()
                training_time += end - start
                adv_time +=  end - start

            else:

                if adversarial_training:
                    start = timer()
                    x_batch_nat, noop_trans = attack.perturb(x_batch, y_batch, sess)
                    end = timer()
                    adv_time +=  end - start

                else:
                    x_batch_nat, noop_trans = x_batch, noop_trans
                    
            nat_dict = {model.x_input: x_batch_nat,
                        model.y_input: y_batch_nat,
                        model.group: id_batch,
                        model.transform: noop_trans,
                        model.is_training: False}

            ################# Outputting/saving weights and evaluations ###############

            nat_acc = -1.0
            acc_grid = -1.0
            avg_xent_grid = -1.0
            saved_weights = 0

            # Compute training accuracy on this minibatch
            train_nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            # Output to stdout
            if epoch_done:
                epoch_time = timer() - start_epoch
                # Average 
                av_acc = acc_sum/it_count

                # ToDo: Log this to file as well 
                
                # Training accuracy over epoch
                print('Epoch {}:    ({})'.format(epoch_count, datetime.now()))
                print('    training natural accuracy {:.4}%'.format(av_acc * 100))
                print('    {:.4} seconds per epoch'.format(epoch_time))

                # Accuracy on entire test set
                test_nat_acc = sess.run(model.accuracy, feed_dict=cifar_eval_dict)

                print('    test set natural accuracy {:.4}%'.format(test_nat_acc * 100))
                # print('    {:.4} seconds for test evaluation'.format(test_time))



                print("example TIME")
                print(adv_time)
                print("train TIME")
                print(train_time)

                ########### Things to do every xxx epochs #############
                # Check if worstof1 eval should be run
                if it_summary == summary_gap - 1 or epoch_count == max_num_epochs - 1:
                    summary = sess.run(nat_summaries, feed_dict=nat_dict)
                    summary_writer.add_summary(summary, global_step.eval(sess))
                    it_summary = 0
                else:
                    it_summary += 1

                if it_easyeval == easyeval_gap - 1 or epoch_count == max_num_epochs - 1:
                    # Evaluation on adv and natural
                    [acc_nat, acc_adv, avg_xent_nat, avg_xent_adv] =  evaluate(model, attack_eval_random, sess, config, "random", data_path, None)
                    # Save in checkpoint
                    chkpt_id = this_repo.create_training_checkpoint(
                        exp_id, training_step=ii, 
                        epoch=epoch_count, 
                        train_acc_nat=nat_acc,
                        test_acc_adv=acc_adv, test_acc_nat=acc_nat,
                        test_loss_adv=avg_xent_adv, 
                        test_loss_nat=avg_xent_nat)

                    it_easyeval = 0
                else:
                    it_easyeval += 1
                    
                startt = timer()
                if it_ckpt == checkpoint_gap - 1 or epoch_count == max_num_epochs - 1:
                    # Create checkpoint id if non-existent
                    if not chkpt_id :
                        chkpt_id = this_repo.create_training_checkpoint(
                            exp_id, training_step=ii, 
                            epoch=epoch_count, 
                            train_acc_nat=train_nat_acc,
                            test_acc_nat=test_nat_acc)

                    # Save checkpoint data (weights)
                    saver.save(sess,
                               os.path.join(model_dir, '{}_checkpoint'.format(chkpt_id)))
                    print(' chkpt saving took {:.4}s '.format(timer()-startt))
                    it_ckpt = 0
                else:
                    it_ckpt += 1
                
                # Set loss sum, it count back to zero
                acc_sum = train_nat_acc
                epoch_done = 0
                epoch_count += 1 
                start_epoch = timer()
                it_count = 1

            else:
                it_count += 1
                acc_sum += train_nat_acc


            # Actual training step
            start = timer()        
            nat_dict[model.is_training] = True
            sess.run(train_step, feed_dict=nat_dict)
            training_time += timer() - start
            train_time += timer() - start


        runtime = time.time() - start_time

        # Do all evaluations in the last step - on grid
        [_, acc_grid, _, avg_xent_grid] = evaluate(model, attack_eval_grid, sess, config, "grid", data_path, eval_summary_writer)

        
        this_repo.mark_experiment_as_completed(
            exp_id, train_acc_nat=nat_acc,
            test_acc_adv=acc_adv, test_acc_nat=acc_nat, 
            test_acc_grid=acc_grid, runtime=runtime)

    return 0

@click.command()
@click.option('--config', default='configs/cifar10_config_stn.json', type=str)
@click.option('--save-root-path', default='/cluster/work/math/fanyang-broglil/CoreRepo', help='path to project root dir')
@click.option('--worstofk', default=None, type=int)
@click.option('--attack-style', default=None, type=str,
              help='Size multipler for original CIFAR dataset')
# ToDo: should be an Array, currently unused
@click.option('--attack-limits', default=None)
@click.option('--lambda-core', default=None)
@click.option('--num-grouped-ids', default=None, type=int)
@click.option('--num-ids', default=None, type=int)
@click.option('--group-size', default=None, type=int)
@click.option('--use-core', default=None, type=bool)
@click.option('--seed', default=None, type=int)

def train_cli(config, save_root_path, worstofk, attack_style, attack_limits,
              lambda_core, num_grouped_ids, num_ids, group_size,use_core,seed):
    '''Train a ResNet on Cifar10.'''
    train(config, save_root_path, worstofk, attack_style, attack_limits,
          lambda_core, num_grouped_ids, num_ids, group_size,use_core,seed)

if __name__ == '__main__':
    train_cli()
