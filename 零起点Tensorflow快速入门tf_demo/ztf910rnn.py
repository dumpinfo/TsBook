#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import pdb
import tensorflow as tf
import tflearn

import numpy as np
import pandas as pd
#from tflearn.data_utils import load_csv

#-----------------

pdb.Restart()

#1
print('\n#1,set.dat')

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6 
X[1,6,:] = 0
X_lengths = [10, 6]

cell = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * 4, state_is_tuple=True)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

#result = tf.contrib.learn.run_n(
result = tflearn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)
