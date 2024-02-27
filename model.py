import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from TuckerLayer import TuckerLayer
from tensorflow.keras.optimizers import Adam
from tensorly.decomposition import tucker
import sys
import os

# FOR REPRODUCIBILITY
def set_seed(s):
  from numpy.random import seed
  seed(s)
  from tensorflow.random import set_seed
  set_seed(s)
#######


modes = (25,25,25)
batch_size = 128
nb_epochs = 25000 # many epochs
nb_missing = 30 # percent of missing data
verbo = 0
log_all = False

try:
  lr = float(sys.argv[1])
except:
  lr = 0.0005

try:
  std_init = float(sys.argv[2])
except:
  std_init = 0.0005 #0.001

try:
  depth = int(sys.argv[3])
except:
  depth = 2

try:
  seed = int(sys.argv[4])
except:
  seed = 123

factor = 100

print(sys.argv)
fig_name = str(lr)+'_'+str(depth)+'_'+str(std_init)+'_'+str(seed)+'_'+str(factor)

set_seed(seed)

########
T, mask, X_train, Y_train, X_test, Y_test = pickle.load(open("data_25-25-25_30_5-5-5.pkl", "rb"))


######## keras callback ######
class CustomCallback(keras.callbacks.Callback):
  def __init__(self, model):
    self.model = model
    self.normsS = []
    self.normsV = []
    for i in range(model.layers[0].core.shape[0]):
     self.normsS.append([])
     self.normsV.append([])
    self.counter = 0
    self.step = 100
    if log_all: self.step = 1

  # during training at each batch
  # compute and store norms of slices and vectors of mode 1
  def on_train_batch_end(self, batch, logs=None):
    c = model.layers[0].core
    if depth > 0: f = model.layers[0].matrices
    if log_all or self.counter % self.step == 0:
       l = []
       for i in range(model.layers[0].core.shape[0]):
           self.normsS[i].append(tf.linalg.norm(c[i,:,:]).numpy())
           if depth > 0: self.normsV[i].append(tf.linalg.norm(f[0][:,i]).numpy())
    self.counter += 1


########## data batch preparation #######

# create batches of (x, y) where x is a tensor of zeroes except a single one at index given in X ; and y the corresponding scalar value given in Y.
def datagenerator(X, Y, batchsize):
    while True:
      start = 0
      end = batchsize

      while start  < len(Y):
          x = X[start:end]
          tx = np.zeros((batchsize,)+modes)
          xx = [(i,)+tuple(w) for i, w in enumerate(x)]
          for i in range(len(x)): tx[xx[i]] = 1.

          y = Y[start:end]
          p = np.random.permutation(len(y))
          start += batchsize
          end += batchsize
          yield tx[p], y[p]

data = datagenerator(X_train, Y_train, batch_size) #train set
data_test = datagenerator(X_test, Y_test, batch_size) #test set

##### model preparation #####

model = keras.models.Sequential()
model.add(TuckerLayer(modes, modes, std_init, depth, seed, factor, input_shape=modes))

model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
print(model.summary())

############# training and test with fit and evaluate

cc = CustomCallback(model)
history = model.fit(data, batch_size=batch_size, steps_per_epoch=int(len(X_train)/batch_size), epochs=nb_epochs, verbose=verbo, callbacks=[cc])

model.evaluate(data_test, steps=int(len(X_test)/batch_size))

########### post training logs #######

# build tensor W from overparameterized deep tucker decomposition

### prepare einsum strings
einmul = ""
s = "ijklmnnopqstuvwx"
for i in range(depth-1):
  einmul += s[i]+s[i+1]+","
einmul = einmul[:-1]
einmul += "->"+s[0]+s[depth-1]

### multiply depth matrices first, then multiply with column vector, then shape to matrices
if depth > 1:
  overparams = []
  for i in range(len(modes)): #modes
    o = []
    for j in range(modes[i]): #columns
      m = tf.einsum(einmul, *(model.layers[0].overparams[i][j]))
      o.append(tf.einsum('ij, j->i', m, model.layers[0].matrices[i][:,j]))
    overparams.append(tf.transpose(tf.convert_to_tensor(o)))
elif depth == 1:
  overparams = model.layers[0].matrices

### get matrices without overparametrization for further logs
if depth >= 1:
  mats = []
  for i in range(len(modes)):
    mats.append(model.layers[0].matrices[i].numpy())
else:
  mats = None

### build tensor W from deep tucker form
s = list("ijklmnnopqstuvwx")
str0 = s[:len(modes)]
str1 = s[:len(modes)]
str2 = list("aa")
W = model.layers[0].core
if depth > 0:
  for i in range(len(modes)):
      str1 = list(str0)
      str1[i] = 'a'
      str2[1] = str0[i]
      einstr = "".join(str0)+","+"".join(str2)+"->"+"".join(str1)
      W = np.einsum(einstr, W, overparams[i])
else:
  W = c
  W.ndim = T.ndim

# compute train/test errors
mask2 = 1 - mask
Tmask = T * mask
Tmask2 = T * mask2
error_train = tf.reduce_sum(tf.square(T*mask - W*mask)).numpy()/len(X_train)
error_test = tf.reduce_sum(tf.square(T*mask2 - W*mask2)).numpy()/len(X_test)
losses = np.array(history.history["loss"])
print("error_train =", error_train)
print("error_test =", error_test)

# get norms of core slices and store numbers of slices above three thresholds
print("Norms of slices of core")
print('')
threshold1 = .1
threshold2 = 1.
threshold3 = 10.
implranks = []
strranks = ""

c = model.layers[0].core
for m in range(len(modes)):
  l = []
  for i in range(modes[m]):
      l.append(tf.linalg.norm(tf.gather(c, i, axis=m)).numpy())
  l = np.array(l)
  print(l)
  print("mode "+str(m+1), len(l[l>threshold1]), len(l[l>threshold2]), len(l[l>threshold3]))
  strranks += str(len(l[l>threshold1]))+','+str(len(l[l>threshold2]))+','+str(len(l[l>threshold3]))+' '
  implranks.append(l)
print('')

# check sparsity of core after full rank HOSVD of W
implranks2 = []
strranks2 = ''
c, f = tucker(W, modes)
for m in range(len(modes)):
  l = []
  for i in range(modes[m]):
      l.append(tf.linalg.norm(tf.gather(c, i, axis=m)).numpy())
  l = np.array(l)
  print(l)
  print("HOSVD mode "+str(m+1), len(l[l>threshold1]), len(l[l>threshold2]), len(l[l>threshold3]))
  strranks2 += str(len(l[l>threshold1]))+','+str(len(l[l>threshold2]))+','+str(len(l[l>threshold3]))+' '
  implranks2.append(l)
print('')

# store logs in pickle file
print('')
print('dat_'+fig_name+'.pkl')
pickle.dump((lr, std_init, depth, factor, error_train, error_test, implranks, implranks2, model.layers[0].core.numpy(), mats, W, cc.normsS, cc.normsV),open("dat_"+fig_name+".pkl", "wb"))

print(lr, std_init, depth, factor, "-", error_train, error_test, strranks, strranks2)
