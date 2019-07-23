#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#-----------------

#1
print('\n#1,set.dat')

rlog='/ailib/log_tmp'
x = tf.constant(2.0, name='input')
w = tf.Variable(0.8, name='weight')
y_model = tf.multiply(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y_model - y_, 2, name='loss')

print('x,',x)
print('w,',w)
print('y_model,',y_model)
print('y_,',y_)
print('loss,',loss)
