#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from time import time as time
import numpy as np
from scipy.stats import wishart
from tensorly.decomposition import tucker
#from tensorly.random import random_cp
import pickle
#import tensorflow as tf

# d: list of dimensions
def build_random_tensor(d):
    A = np.random.standard_normal(d)
    A = A/np.max(np.abs(A))
    return A

# rate: percent of zeros in mask
# d: list of dimensions
def build_random_mask(rate, d):
    A = np.random.uniform(0, 1, d)
    nb_holes = int(np.prod(d)*rate/100.)
    for i in range(nb_holes):
      A[A == A.max()] = 0
    A[A > 0] = 1
    return A


# launch python make_data.py <test_split_in_percent>
# ie: python make_data.py 30
# output a pickle file with the original tensor as well as train/test split.,
if __name__ == '__main__':

    dim = (10,10,10) # dim of tensor to generate
    rank = (3,3,3) # tucker rank of tensor
    rate = int(sys.argv[1]) # percent of test data
    
    A = build_random_tensor(dim)
    
    # do full rank tucker factorizatiob
    c, f = tucker(A, dim)
        
    # normalize slices of core
    BB = A.numpy()
    #AA /= np.linalg.norm(AA)
    for m in range(len(modes)):
      for i in range(modes[m]):
        if m == 0: BB[i,:,:] /= np.linalg.norm(A[i,:,:])
        elif m == 1: BB[:,i,:] /= np.linalg.norm(A[:,i,:])
        elif m == 2: BB[:,:,i] /= np.linalg.norm(A[:,:,i])
    A = BB

    # print norm of slices of every modes
    for m in range(len(dim)):
      print([np.linalg.norm(np.take(A, i, axis=m)) for i in range(dim[m])])

    c, f = tucker(A, rank) # do low rank tucker factorizatiob
   
    # reconstruction with einsum operations
    s = list("ijklmnnopqstuvwx")
    str0 = s[:len(dim)]
    str1 = s[:len(dim)]
    str2 = list("aa")
    W = c
    for i in range(len(dim)):
      str1 = list(str0)
      str1[i] = 'a'
      str2[1] = str0[i]
      einstr = "".join(str0)+","+"".join(str2)+"->"+"".join(str1)
      W = np.einsum(einstr, W, f[i])
      print(einstr)
      
    # print norm of slices of every modes
    for m in range(len(dim)):
      print([np.linalg.norm(np.take(W, i, axis=m)) for i in range(dim[m])])
      
    # create random tensor of zrtpes and rate percent ones
    mask = build_random_mask(rate, dim)
    Y = W * mask
    Y[Y == -0] = 0

    #create pairs (indices ; value) where indices is a coordintate within tensor W
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for x, y in np.ndenumerate(Y):
      if y == 0:
        X_test.append(x)
        Y_test.append(W[x])
      else:
        X_train.append(x)
        Y_train.append(y)

    # pack enerything with numpy and pickle
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)[:, np.newaxis]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)[:, np.newaxis]

    pickle.dump((W, mask, X_train, Y_train, X_test, Y_test), open("data_"+str(dim).replace(' ',
    '').replace('(','').replace(')','').replace(',','-')+"_"+str(rate)+"_"+str(rank).replace(' ',
    '').replace('(','').replace(')','').replace(',','-')+".pkl", "wb"))

