#!/usr/bin/env python
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal
from tensorly.decomposition import tucker # EXPE2
from scipy.linalg import qr # EXPE2


# Core initializer from random normal with given parameters.
class CoreRandomNormal(tf.keras.initializers.Initializer):
  def __init__(self, mean, stddev, rank):
    self.mean = mean
    self.stddev = stddev
    self.rank = rank

  def __call__(self, shape, dtype=None, **kwargs):
      rnd = np.random.normal(self.mean, self.stddev, shape)
      return rnd

# Overparam matrices initialization with norm equal to corresponding slice
class NormRandomNormal(tf.keras.initializers.Initializer):
  def __init__(self, mean, stddev, core, idx, jdx):
    self.mean = mean
    self.stddev = stddev
    self.core = core
    self.idx = idx
    self.jdx = jdx

  def __call__(self, shape, dtype=None, **kwargs):
      rnd = np.random.normal(self.mean, self.stddev, shape)
      rnd = rnd * np.linalg.norm(tf.gather(self.core, self.jdx, axis=self.idx).numpy()) / np.linalg.norm(rnd)
      print(np.linalg.norm(tf.gather(self.core, self.jdx, axis=self.idx).numpy()), np.linalg.norm(rnd))
      return rnd

# Column of factor matrices initialization with norm equal to corresponding slice
class ColNormRandomNormal(tf.keras.initializers.Initializer):
  def __init__(self, mean, stddev, core, idx):
    self.mean = mean
    self.stddev = stddev
    self.core = core
    self.idx = idx

  def __call__(self, shape, dtype=None, **kwargs):
      rnd = np.random.normal(self.mean, self.stddev, shape)
      for j in range(rnd.shape[1]):
        rnd[:,j] = rnd[:,j] * np.linalg.norm(tf.gather(self.core, j, axis=self.idx).numpy()) / np.linalg.norm(rnd[:,j])
        print(np.linalg.norm(tf.gather(self.core, j, axis=self.idx).numpy()), np.linalg.norm(rnd[:,j])) #self.matrices[i][:,j])
      return rnd

# Class TuckerLayer implements overparameterized deep tucker model for any size of tensors and multiranks
class TuckerLayer(Layer):
    def __init__(self, rank, modes, std_init, depth, seed, ff, **kwargs):
        self.rank = rank # list of ints
        self.modes = modes # list of dims
        self.std_init = std_init
        self.depth = depth
        self.init_mode = 1
        self.seed = seed
        self.mean = 0.0
        self.ff = ff # factor

        
        # prepare some einsum strings..
        s = "ijklmnnopqstuvwx"
        self.einmul = ""
        for i in range(depth-1):
          self.einmul += s[i]+s[i+1]+","
        self.einmul = self.einmul[:-1]
        self.einmul += "->"+s[0]+s[depth-1]

        str0 = s[:len(self.modes)]
        str1 = s[:len(self.modes)]
        str2 = 'aa'
        self.einmodes = []
        for i in range(len(self.modes)):
          str1 = list(str0)
          str1[i] = 'a'
          str1 = "".join(str1)
          str2 = list(str2)
          str2[1] = str0[i]
          str2 = "".join(str2)
          self.einmodes.append(str0+','+str2+'->'+str1)
          print(str0+','+str2+'->'+str1)
        self.strmodes = s[:len(self.modes)]

        super(TuckerLayer, self).__init__(**kwargs)

    # define and initialize parameters to train
    def build(self, input_shape):
        l = []
        ll = []

        # bigger initialization for deeper models for much faster convergence
        d = 1.
        if self.depth > 0: d = self.depth
        std_init = self.std_init*d*np.log(d+1)
        
        # Core tensor
        self.core = self.add_weight(shape=self.rank, initializer=CoreRandomNormal(0, std_init, self.rank), trainable=True)

        if self.depth > 0:
          # Mode matrices
          for i in range(len(self.modes)):
            l.append(self.add_weight(shape=(self.rank[i], self.modes[i]), initializer=ColNormRandomNormal(0, std_init, self.core, i), trainable=True))

          # Overparams matrices
          for i in range(len(self.modes)): #modes
            lll = []
            for j in range(self.modes[i]): #columns
              llll = []
              for k in range(self.depth - 1): #depth
                llll.append(self.add_weight(shape=(self.rank[i], self.rank[i]), initializer=NormRandomNormal(0, std_init, self.core, i, j), trainable=True))
              lll.append(llll)
            ll.append(lll)
          self.matrices = l
          self.overparams = ll

        super(TuckerLayer, self).build(input_shape)

    def reinitCore(self):
      self.core = tf.Variable(np.random.normal(self.mean, self.std_init, self.core.shape))

    # implement forward pass for a batch of x
    def call(self, x):

        # matrix products with columns of mode matrices
        if self.depth > 1:
          overparams = []
          for i in range(len(self.modes)): #modes
            o = []
            for j in range(self.modes[i]): #columns
              m = tf.einsum(self.einmul, *(self.overparams[i][j]))
              o.append(tf.einsum('ij, j->i', m, self.matrices[i][:,j]))
            overparams.append(tf.transpose(tf.convert_to_tensor(o)))
        elif self.depth == 1: # TEST TEST
          overparams = self.matrices

        # k-mode products with core tensor
        core = self.core
        if self.depth > 0 and True:
          for i in range(len(self.modes)):
            core = tf.einsum(self.einmodes[i], core, overparams[i]) #self.matrices[i])

        # return tensor values for batch of x samples
        return tf.einsum('b'+self.strmodes+', '+self.strmodes+'->b', x, core)

    # ouput shape is the same as input shape
    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

