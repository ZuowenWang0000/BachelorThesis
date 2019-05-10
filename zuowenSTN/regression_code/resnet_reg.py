# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from spatial_transformer import transformer


# This ResNet is modified for regression task


class Model(object):
  """ResNet model."""

  def __init__(self, config, num_ids, differentiable, adversarial_ce=False):
    """ResNet constructor.
    """
    self._build_model(config, config.filters, num_ids, differentiable,
                      adversarial_ce,
                      pad_mode=config.pad_mode,
                      pad_size=config.pad_size)

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, config, filters, num_ids, differentiable=False,
                   adversarial_ce=False,
                   pad_mode='CONSTANT',
                   pad_size=32):
    """Build the core model within the graph."""
    with tf.variable_scope('input'):

      self.group = tf.placeholder(tf.int32, [None], name="group")
      self.num_ids = num_ids

      self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
      #MODIFIED now y_input is a three dimensional label
      # self.y_input = tf.placeholder(tf.int64, shape=None)
      self.y_input = tf.placeholder(tf.float32, shape=[None, config.n_regression_labels]) #should be 3

      self.transform = tf.placeholder(tf.float32, shape=[None, 3])
      trans_x, trans_y, rot = tf.unstack(self.transform, axis=1)
      rot *= np.pi / 180 # convert degrees to radians

      self.is_training = tf.placeholder(tf.bool)

      x = self.x_input

      self.before_reflect_x = x[0:4]

      x = tf.pad(x, [[0,0], [16,16], [16,16], [0,0]], pad_mode)

      if not differentiable:
        # For spatial non-PGD attacks: rotate and translate image
        ones = tf.ones(shape=tf.shape(trans_x))
        zeros = tf.zeros(shape=tf.shape(trans_x))
        trans = tf.stack([ones,  zeros, -trans_x,
                          zeros, ones,  -trans_y,
                          zeros, zeros], axis=1)
        x = tf.contrib.image.rotate(x, rot, interpolation='BILINEAR')
        x = tf.contrib.image.transform(x, trans, interpolation='BILINEAR')
      else:
        print("in training localization net, this section should not be entered, go check error\n")
        # for spatial PGD attacks need to use diffble transformer
        theta = tf.stack([tf.cos(rot), -tf.sin(rot), trans_x/64,
                          tf.sin(rot),  tf.cos(rot), trans_y/64], axis=1)
        x = transformer(x, theta, (64,64))
      x = tf.image.resize_image_with_crop_or_pad(x, pad_size, pad_size)

      self.reflect_x = x[0:4]

      # everything below this point is generic (independent of spatial attacks)
      self.x_image = x
      x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)

      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in range(1, config.resnet_depth_n):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    # uncomment to add and extra fc layer
    #with tf.variable_scope('unit_fc'):
    #  self.pre_softmax = self._fully_connected(x, 1024)
    #  x = self._relu(x, 0.1)

    # with tf.variable_scope('logit'):
    #   self.pre_softmax = self._fully_connected(x, config.n_classes)

    # self.prediction = tf.argmax(self.pre_softmax, 1)
    # self.correct_prediction = tf.equal(self.prediction, self.y_input)
    # self.num_correct = tf.reduce_sum(
    #     tf.cast(self.correct_prediction, tf.int64))
    # self.accuracy = tf.reduce_mean(
    #     tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('prediction'):
      self.prediction = self._fully_connected(x, config.n_regression_labels)

    with tf.variable_scope('error'):
      self.err = tf.abs(self.prediction - self.y_input)
      self.err_x, self.err_y, self.err_rot = tf.unstack(self.err, axis = 1)
      self.err_x_rel = tf.abs(self.err_x / self.y_input[:, 0])
      self.err_y_rel = tf.abs(self.err_y / self.y_input[:, 1])
      self.err_rot_rel = tf.abs(self.err_rot / self.y_input[:, 2])
      worst_err_temp = tf.maximum(self.err_x_rel, self.err_y_rel)
      worst_err_temp = tf.maximum(worst_err_temp, self.err_rot_rel)
      self.err_worst_rel = worst_err_temp
    with tf.variable_scope('avg_abs_error'):
      self.avg_abs_err_transX = tf.reduce_mean(tf.abs(self.err[:, 0]))
      self.avg_abs_err_transY = tf.reduce_mean(tf.abs(self.err[:, 1]))
      self.avg_abs_err_rot = tf.reduce_mean(tf.abs(self.err[:, 2]))
    with tf.variable_scope('avg_rel_error'):
      self.avg_rel_err_transX = tf.reduce_mean(tf.abs(self.err[:, 0]/self.y_input[:, 0]))
      self.avg_rel_err_transY = tf.reduce_mean(tf.abs(self.err[:, 1]/self.y_input[:, 1]))
      self.avg_rel_err_rot = tf.reduce_mean(tf.abs(self.err[:, 2]/self.y_input[:, 2]))
      self.avg_worst_err = tf.reduce_mean(tf.abs(self.err_worst_rel))



    with tf.variable_scope('costs'):
      if config.loss_type == 1:
        #  L1 loss
        self.reg_loss = tf.reduce_sum(tf.losses.absolute_difference(
            self.y_input, #labels, 3 dimensional
            self.prediction) #prediction
        )
      elif config.loss_type == 2:
        # L2 loss
        self.reg_loss = tf.reduce_sum(tf.losses.mean_squared_error(
            self.y_input, #labels, 3 dimensional
            self.prediction) #prediction
        )
      elif config.loss_type == 3:
        # L1 rel loss
        self.reg_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(
          (self.y_input-self.prediction)/self.y_input, axis=0)))
      elif config.loss_type == 4:
        self.reg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
          (self.y_input-self.prediction)/self.y_input), axis=0))

      self.weight_decay_loss = self._decay()
      #Core penalty
      #Not in use for two-stage training
      # self.core_loss = self._CoRe()
      # self.core_loss2 = self._CoRe_2tensors()


  def _CoRe(self):
    partition_y = tf.dynamic_partition(self.pre_softmax, self.group,
                                       self.num_ids)
    part_var = [tf.reduce_sum(tf.nn.moments(partition,axes=[0])[1])
                for partition in partition_y]
    countfact_loss = tf.reduce_sum(part_var)/self.num_ids
    return countfact_loss


  def _CoRe_2tensors(self):
    # assuming first num_ids are natural examples; second num_ids
    # are adversarially transformed ones
    natural_examples = tf.gather(self.pre_softmax,
      tf.cast(tf.range(tf.shape(self.pre_softmax)[0] - self.num_ids), tf.int32))
    adversarial_examples = tf.gather(self.pre_softmax,
      tf.cast(tf.range(self.num_ids, tf.shape(self.pre_softmax)[0]), tf.int32))
    group_vars = tf.reduce_sum(
      tf.square(natural_examples-adversarial_examples), axis=1)
    # we divide by 4 to match the variance computation from above
    # a factor of 1/2 is required anyways; the second factor of 1/2
    # corresponds to the biased estimate of the variance
    # which is also used in tf.nn.moments*()
    countfact_loss = tf.reduce_mean(group_vars)/4.
    return countfact_loss


  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=self.is_training)

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') >= 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
