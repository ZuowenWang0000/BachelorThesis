"""
Implementation of a spatial attack.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product, repeat
import random

import tensorflow as tf
import numpy as np

from pgd_attack import LinfPGDAttack, SpatialPGDAttack

class SpatialAttack:
  def __init__(self, model, config, method=None, worstofk=None,
               attack_limits=None, fo_epsilon=2.0, fo_step_size=2.,
               fo_num_steps=5):
    self.model = model
    self.grid_store = []

    self.simple_train = config.simple_train

    if config.use_linf:
        self.linf_attack = LinfPGDAttack(
            model, config, fo_epsilon, fo_step_size, fo_num_steps)
    else:
        self.linf_attack = None

    self.use_spatial = config.use_spatial
    if config.use_spatial:
      # Attack method
        if method == None:
          self.method = config.spatial_method
        else:
          self.method = method

        # Attack parameters
        if attack_limits == None:
          self.limits = config.spatial_limits
        else:
          self.limits = attack_limits

        if config.only_rotation:
            self.limits = [0,0,self.limits[2]]

        if config.only_translation:
            self.limits = [self.limits[0],self.limits[1],0]

        # Attack method parameters
        if self.method == 'grid':
            self.granularity = config.grid_granularity
        elif self.method == 'random':
          if worstofk == None:
            self.random_tries = config.random_tries
          else:
            self.random_tries = worstofk
        elif self.method == 'fo':
            self.fo_attack = SpatialPGDAttack(
                model, config, fo_epsilon, fo_step_size, fo_num_steps)
        else:
            raise NotImplementedError


  def perturb(self, x_nat, y, sess):
      if not self.use_spatial:
          t = np.zeros([len(x_nat), 3])
          if self.linf_attack:
              x = self.linf_attack.perturb(x_nat, y, sess, trans=t)
          else:
              x = x_nat
          return x, t
      if self.method == 'grid':
          return self.perturb_grid(x_nat, y, sess, -1)
      elif self.method == 'fo':
          return self.fo_attack.perturb(x_nat, y, sess)
      else: # random
          return self.perturb_grid(x_nat, y, sess, self.random_tries)

  def simple_train_perturb(self, x_nat):
    n = len(x_nat)
    x_nat_extended = np.concatenate((x_nat, x_nat, x_nat, x_nat))
    grid = [(42, 42, 42)]   # dummy grid
    for tx, ty, r in grid:
        # randomize each example separately
        t = np.stack((np.random.uniform(-l, l,  4*n) for l in self.limits),
                         axis=1)

        # #notice that now we pass the transformation parameters as y labels
        # curr_dict = {self.model.x_input: x_nat,
        #              self.model.y_input: t,
        #              self.model.is_training: False,
        #              self.model.transform: t}
    return x_nat_extended, t
    # return x_nat, t

  def eval_perturb(self, x_nat):
    n = len(x_nat)
    # x_nat_extended = np.concatenate((x_nat))
    grid = [(42, 42, 42)]   # dummy grid
    for tx, ty, r in grid:
        # randomize each example separately
        t = np.stack((np.random.uniform(-l, l, n) for l in self.limits),
                         axis=1)

        # #notice that now we pass the transformation parameters as y labels
        # curr_dict = {self.model.x_input: x_nat,
        #              self.model.y_input: t,
        #              self.model.is_training: False,
        #              self.model.transform: t}
    return t

  # def simple_train_perturb(self, x_nat):
  #   n = len(x_nat)
  #   x_nat_extended = x_nat
  #   grid = [(42, 42, 42)]   # dummy grid
  #   for tx, ty, r in grid:
  #       # randomize each example separately
  #       t = np.stack((np.random.uniform(-l, l, 3 * n) for l in self.limits),
  #                        axis=1)
  #
  #       # #notice that now we pass the transformation parameters as y labels
  #       # curr_dict = {self.model.x_input: x_nat,
  #       #              self.model.y_input: t,
  #       #              self.model.is_training: False,
  #       #              self.model.transform: t}
  #   return x_nat_extended, t


  def perturb_grid(self, x_nat, y, sess, random_tries=-1):
    n = len(x_nat)
    if random_tries > 0:
        # subsampling this list from the grid is a bad idea, instead we
        # will randomize each example from the full continuous range
        grid = [(42, 42, 42) for _ in range(random_tries)] # dummy list
    else: # exhaustive grid
        grid = product(*list(np.linspace(-l, l, num=g)
                             for l, g in zip(self.limits, self.granularity)))

    worst_x = np.copy(x_nat)
    worst_t = np.zeros([n, 3])
    max_xent = np.zeros(n)
    all_correct = np.ones(n).astype(bool)

    for tx, ty, r in grid:
        if random_tries > 0:
            # randomize each example separately
            t = np.stack((np.random.uniform(-l, l, n) for l in self.limits),
                         axis=1)
        else:
            t = np.stack(repeat([tx, ty, r], n))

        if self.linf_attack:
            x = self.linf_attack.perturb(x_nat, y, sess, trans=t)
        else:
            x = x_nat


        curr_dict = {self.model.x_input: x,
                     self.model.y_input: y,
                     self.model.is_training: False,
                     self.model.transform: t}

        cur_xent, cur_correct = sess.run([self.model.y_xent,
                                          self.model.correct_prediction],
                                         feed_dict = curr_dict) # shape (bsize,)
        cur_xent = np.asarray(cur_xent)
        cur_correct = np.asarray(cur_correct)

        # Select indices to update: we choose the misclassified transformation
        # of maximum xent (or just highest xent if everything else if correct).
        idx = (cur_xent > max_xent) & (cur_correct == all_correct)
        idx = idx | (cur_correct < all_correct)
        max_xent = np.maximum(cur_xent, max_xent)
        all_correct = cur_correct & all_correct

        idx = np.expand_dims(idx, axis=-1) # shape (bsize, 1)
        worst_t = np.where(idx, t, worst_t) # shape (bsize, 3)

        idx = np.expand_dims(idx, axis=-1)
        idx = np.expand_dims(idx, axis=-1) # shape (bsize, 1, 1, 1)
        worst_x = np.where(idx, x, worst_x,) # shape (bsize, 32, 32, 3)

    return worst_x, worst_t
