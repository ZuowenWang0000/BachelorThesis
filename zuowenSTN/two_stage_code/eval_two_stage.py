"""
Evaluation of a given checkpoint in the standard and adversarial sense.  Can be
called as an infinite loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time
import copy

import numpy as np
import tensorflow as tf
from tqdm import trange

import cifar10_input
import cifar100_input
import svhn_input
import resnet
import resnet_reg
import vgg
# import twostage
from spatial_attack import SpatialAttack
import utilities
import click


def simple_transform(x, transform, reg_pad_mode):
    trans_x, trans_y, rot = tf.unstack(transform, axis=1)
    rot *= np.pi / 180  # convert degrees to radians
    ones = tf.ones(shape=tf.shape(trans_x))
    zeros = tf.zeros(shape=tf.shape(trans_x))
    trans = tf.stack([ones, zeros, -trans_x,
                      zeros, ones, -trans_y,
                      zeros, zeros], axis=1)
    x = tf.pad(x, [[0, 0], [16, 16], [16, 16], [0, 0]], reg_pad_mode)
    x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
    x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
    # crop back to standard cifar10 image
    x = tf.image.resize_image_with_crop_or_pad(x, 32, 32)
    return x

# A function for evaluating a single checkpoint
def evaluate(model, attack, sess, config, attack_type, data_path,reg_model_path,
             summary_writer=None, eval_on_train=False):
    num_eval_examples = config.eval.num_eval_examples
    eval_batch_size = config.eval.batch_size

    if config.data.dataset_name == "cifar-10":
        data_iterator = cifar10_input.CIFAR10Data(data_path)
    elif config.data.dataset_name == "cifar-100":
        data_iterator = cifar100_input.CIFAR100Data(data_path)
    elif config.data.dataset_name == "svhn":
        data_iterator = svhn_input.SVHNData(data_path)

    else:
        raise ValueError("Unknown dataset name.")

    global_step = tf.train.get_or_create_global_step()
    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in trange(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      if eval_on_train:
        x_batch = data_iterator.train_data.xs[bstart:bend, :]
        y_batch = data_iterator.train_data.ys[bstart:bend]
      else:
        x_batch = data_iterator.eval_data.xs[bstart:bend, :]
        y_batch = data_iterator.eval_data.ys[bstart:bend]

      noop_trans = np.zeros([len(x_batch), 3])
      if config.eval.adversarial_eval:
          # print("PERTURBED!")
          x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)
      else:
          x_batch_adv, adv_trans = x_batch, noop_trans

      ori_trans = adv_trans
      # ori_trans_x = ori_trans[:0]
      # ori_trans_y = ori_trans[:1]
      # ori_rot = ori_trans[:2]
      ori_trans_x, ori_trans_y, ori_rot = tf.unstack(tf.cast(adv_trans, tf.float32), axis=1)
      adv_trans = tf.cast(adv_trans, tf.float32)
      x_batch_adv_transformed = simple_transform(x_batch_adv, adv_trans, config.model.reg_pad_mode)
      x_batch_adv_transformed = sess.run(x_batch_adv_transformed)
      # utilities.show_image_batch(x_batch_adv_transformed, title='adv')
      # print(ori_trans_x.shape)
      # print(ori_trans_y.shape)
      # print(ori_rot.shape)
      # ones = tf.ones(shape=tf.shape(ori_trans_x))
      # zeros = tf.zeros(shape=tf.shape(ori_trans_x))
      # trans = tf.stack([ones, zeros, -ori_trans_x,
      #                   zeros, ones, -ori_trans_y,
      #                   zeros, zeros], axis=1)
      # temp = tf.contrib.image.transform(x_batch_adv, trans, interpolation='BILINEAR')
      # x_batch_adv_transformed = tf.contrib.image.rotate(temp, ori_rot, interpolation='BILINEAR')


      # use regression model to turn back peturbed adversarial image (for evaluation)
      g_temp = tf.Graph()

      with tf.Session(graph = g_temp) as sess2:
          print("sub session")
          id_batch = np.arange(0, num_ids, 1, dtype="int32")

          with g_temp.as_default():
              # Restore the checkpoint of the regression model
              metapath = reg_model_path + ".meta"
              saver = tf.train.import_meta_graph(metapath)
              saver.restore(sess2, reg_model_path)

              # init.run()
              # # load model_reg
              model_reg = resnet_reg.Model(config.model, num_ids, diffable,
                                   config.training.adversarial_ce)

              graph=tf.get_default_graph()
              # restore tensors
              group = graph.get_tensor_by_name("input/group:0")
              x_input = graph.get_tensor_by_name("input/Placeholder:0")
              y_input = graph.get_tensor_by_name("input/Placeholder_1:0")
              transform = graph.get_tensor_by_name("input/Placeholder_2:0")
              is_training = graph.get_tensor_by_name("input/Placeholder_3:0")

              prediction = graph.get_tensor_by_name("prediction/xw_plus_b:0")

              # This is just a placeholder, since inference in reg model does not need label
              y_batch_placeholder = np.zeros([len(x_batch), 3]).astype(np.float)
              x_batch_adv = x_batch_adv.astype(np.float)
              x_batch = x_batch.astype(np.float)

              # the original x_batch_adv is not transformed yet, so we need to transform
              # before we feed the batch into the model



              regression_feed = {x_input: x_batch_adv_transformed,
                                 y_input: y_batch_placeholder,
                                 group: id_batch,
                                 transform: noop_trans,
                                 is_training: False}
              predictions = sess2.run(prediction, feed_dict=regression_feed)

              # we also feed the test batch for natural accuracy into the regression mode
              # to check how much will it degrade the nat.acc
              nat_reg_feed = {x_input: x_batch_adv,
                          y_input: y_batch_placeholder,
                          group: id_batch,
                          transform: noop_trans,
                          is_training: False
              }
              nat_predictions = sess2.run(prediction, feed_dict=nat_reg_feed)
              # using the same style of transformation: for nat. should be noop_trans - nat_predictions
              nat_attack = noop_trans - nat_predictions

              # op = sess2.graph.get_operations()
              # for m in op:
              #     print(m.name)



              # print(transPara.shape)
              # trans_x = transPara[:1]
              # trans_y = transPara[:2]
              # rot = transPara[:3]

              trans_x, trans_y, rot = tf.unstack(predictions, axis=1)
              ones = tf.ones(shape=tf.shape(trans_x))
              zeros = tf.zeros(shape=tf.shape(trans_x))

              print("check shapes")
              print(predictions.shape)
              print(ori_trans.shape)



              print("original attack")
              print("trans X: {}".format(sess.run(ori_trans_x)[0:4]))
              print("trans Y: {}".format(sess.run(ori_trans_y)[0:4]))
              print("rot angle: {}".format(sess.run(ori_rot)[0:4]))

              print("\npredicted attack")
              print("trans X: {}".format(sess2.run(trans_x)[0:4]))
              print("trans Y: {}".format(sess2.run(trans_y)[0:4]))
              print("rot angle: {}".format(sess2.run(rot)[0:4]))

              remain_diff = ori_trans - predictions


              diff_x, diff_y, diff_rot = tf.unstack(remain_diff, axis=1)
              print("\nremain_diff")
              print("trans X: {}".format(sess2.run(diff_x)[0:4]))
              print("trans Y: {}".format(sess2.run(diff_y)[0:4]))
              print("rot angle: {}".format(sess2.run(diff_rot)[0:4]))

              x = x_batch_adv_transformed
              # print(ones.shape)
              trans_1 = tf.stack([ones, zeros, trans_x,
                                zeros, ones, trans_y,
                                zeros, zeros], axis=1)

              x = tf.contrib.image.transform(x, trans_1, interpolation='BILINEAR')
              x = tf.contrib.image.rotate(x, -rot*np.pi/180.0, interpolation='BILINEAR')


              de_adv = sess2.run(x)
              # utilities.show_image_batch(de_adv, title = 'de_adv')
              dict_nat = {model.x_input: x_batch,
                          model.y_input: y_batch,
                          # model.transform: noop_trans,
                          model.transform: nat_attack,
                          model.is_training: False}
              # we the transformation is done in the model part, Thus we use the adversarial_attack_para - prediction
              # as the new model.transform,  if it's perfectly predicted, then it should be close to zero
              dict_adv = {model.x_input: x_batch_adv,
                          model.y_input: y_batch,
                          model.transform: remain_diff,
                          model.is_training: False}
              # print("entered new session")

      cur_corr_nat, cur_xent_nat = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run([model.num_correct, model.xent],
                                            feed_dict = dict_adv)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    if summary_writer:
        summary = tf.Summary(value=[
              tf.Summary.Value(tag='xent_adv_eval', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent_nat_eval', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='xent_adv', simple_value= avg_xent_adv),
              tf.Summary.Value(tag='xent_nat', simple_value= avg_xent_nat),
              tf.Summary.Value(tag='accuracy_adv_eval', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy_nat_eval', simple_value= acc_nat),
              tf.Summary.Value(tag='accuracy_adv', simple_value= acc_adv),
              tf.Summary.Value(tag='accuracy_nat', simple_value= acc_nat)])
        summary_writer.add_summary(summary, global_step.eval(sess))

    step = global_step.eval(sess)
    print('Eval at step: {}'.format(step))
    print('  Adversary: ', attack_type)
    print('  natural: {:.2f}%'.format(100 * acc_nat))
    print('  adversarial: {:.2f}%'.format(100 * acc_adv))
    print('  avg nat xent: {:.4f}'.format(avg_xent_nat))
    print('  avg adv xent: {:.4f}'.format(avg_xent_adv))

    return [100 * acc_nat, 100 * acc_adv, avg_xent_nat, avg_xent_adv]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Eval script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default="configs/christinaconfig_cifar10_spatial_eval.json", required=False)
    parser.add_argument('--save_root_path', type=str,
                        help='path to repo dir',
                        default='./noAdvCoreRepo/logdir', required=False)
    parser.add_argument('--exp_id_list', type=str, nargs='+',
                        default=['WXgywpiVKs_1757588'])
    parser.add_argument('--eval_on_train', type=int,
                        help='flag whether to use training or test images',
                        default=0, required=False)
    parser.add_argument('-s', '--save_filename', type=str,
                        help='path to plots folder',
                        default='test.json', required=False)
    parser.add_argument('--linf_attack', type=int,
                        help='path to plots folder',
                        default=0, required=False)
    parser.add_argument('--reg_model_path', type=str,
              default='../regression_code/regRepo/logdir/M7LJmbVxq9_1807632/checkpoint-180000')

    args = parser.parse_args()
    config_dict = utilities.get_config(args.config)
    dataset = config_dict['data']['dataset_name']
    # setting up save folders
    split = 'train' if args.eval_on_train else 'test'
    print(args.exp_id_list)
    save_folder = os.path.join(args.save_root_path,
        'additional_evals_{}'.format(dataset))
    os.makedirs(save_folder, exist_ok=True)
    save_filename = os.path.join(save_folder,
        '{}_{}_{}'.format(dataset, split, args.save_filename))

    if args.eval_on_train:
        if dataset == 'cifar-10' or dataset == 'cifar-100':
            config_dict['eval']['num_eval_examples'] = 50000
        elif dataset == 'svhn':
            config_dict['eval']['num_eval_examples'] = 73257
        else:
            raise NotImplementedError

    config_dict_copy = copy.deepcopy(config_dict)
    out_dict = {}
    out_dict['hyperparameters'] = config_dict_copy
    config = utilities.config_to_namedtuple(config_dict)

    reg_model_path = args.reg_model_path

    # num_ids in model does not matter for eval
    num_ids = 64
    model_family = config.model.model_family
    if model_family == "resnet":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = resnet.Model(config.model, num_ids, diffable)
    elif model_family == "vgg":
        if config.attack.use_spatial and config.attack.spatial_method == 'fo':
            diffable = True
        else:
            diffable = False
        model = vgg.Model(config.model, num_ids, diffable)

    global_step = tf.train.get_or_create_global_step()
    if args.linf_attack:
        attack_eval = SpatialAttack(model, config.attack, 'fo', 1,
            config.attack.spatial_limits,
            config.attack.epsilon,
            config.attack.step_size,
            config.attack.num_steps)
    else:
        # TODO used to be grid
        # attack_eval_random = SpatialAttack(model, config.attack, 'random')
        attack_eval = SpatialAttack(model, config.attack, 'grid')

    saver = tf.train.Saver()

    for id in args.exp_id_list:
        print('id :{}'.format(id))
        out_dict[id] = {}
        model_dir = '%s/logdir/%s' % ('./noAdvCoreRepo', id)
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt is None:
            print('No checkpoint found.')
        else:
            with tf.Session() as sess:
                # Restore the checkpoint
                saver.restore(sess,
                    os.path.join(model_dir,
                                 ckpt.model_checkpoint_path.split("/")[-1]))
                print("entering evaluate!")
                [acc_nat, acc_grid, _, _] = evaluate(
                    model, attack_eval, sess, config, 'grid',
                    config.data.data_path, reg_model_path, eval_on_train=args.eval_on_train)
                out_dict[id]['{}_grid_accuracy'.format(split)] = acc_grid
                out_dict[id]['{}_nat_accuracy'.format(split)] = acc_nat

                # [acc_nat, acc_grid, _, _] = evaluate(
                #     model, attack_eval_random, sess, config, 'random',
                #     config.data.data_path, eval_on_train=args.eval_on_train)

                # save results
                with open(save_filename, 'w') as result_file:
                    json.dump(out_dict, result_file, sort_keys=True, indent=4)

    grid_accuracy = []
    nat_accuracy = []
    for key in out_dict:
        if key != 'hyperparameters':
            grid_accuracy.append(out_dict[key]['{}_grid_accuracy'.format(split)])
            nat_accuracy.append(out_dict[key]['{}_nat_accuracy'.format(split)])

    out_dict['{}_grid_accuracy_summary'.format(split)] = (np.mean(grid_accuracy),
        np.std(grid_accuracy))
    out_dict['{}_nat_accuracy_summary'.format(split)] = (np.mean(nat_accuracy),
        np.std(nat_accuracy))

    with open(save_filename, 'w') as result_file:
        json.dump(out_dict, result_file, sort_keys=True, indent=4)

