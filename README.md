# deeptuckerimplicit
Code for Implicit Regularization in Deep Tucker Factorization: Low-Rankness via Structured Sparsity published at AISTATS 2024

3 files are provided:

- make_data.py: script to build synthetic data that creates a random tensor of size (25,25,25) with tucker multirank (5,5,5). One argument: size in percent of test set.

- TuckerLayer.py: tensorflow/keras code that implements overparameterized deep tucker layer. Edit variable nb_blocs to reprocuce experiments.

- model.py : main file to launch experiments. Default arguments are ok

# Launch
$ python make_data.py 30
$ python model.py <learning rate> <std_init> <depth> <seed>
