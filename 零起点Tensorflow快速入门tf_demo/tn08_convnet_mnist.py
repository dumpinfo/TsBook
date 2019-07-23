#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

MNIST手写数字图像识别
CNNcnn卷积神经网络算法
使用TFlearn简化接口 


@from:
tflearn.org
'''

import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import tensorflow as tf

#------------------

#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)

#2
print('\n#2,set MNIST输入数据')
X, Y, testX, testY = mnist.load_data(data_dir='data/mnist/',one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

#3.1
print('\n#3.1,构建线性回归神经网络模型')
#3.1a
# 设置输入数据
network = input_data(shape=[None, 28, 28, 1], name='input')

#3.1b
# CNN卷积运算
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")

#3.1c
# 最大池化运算
network = max_pool_2d(network, 2)

#3.1d
# 局部响应归一化运算
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)

#3.1e
# 全连接运算
network = fully_connected(network, 128, activation='tanh')

#3.1f
# dropout丢弃层运算 
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')

#3.1g
# 回归运算
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

#3.2
print('\n#3.2,使用内置的DNN神经网络模型，设置数据目录：summary日志')
model = tflearn.DNN(network, tensorboard_verbose=0,tensorboard_dir=rlog)

#4
print('\n#4,开始训练模型')
model.fit({'input': X}, {'target': Y}, n_epoch=5,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, #Snapshot (save & evalaute) model every 100 steps.
           show_metric=True, run_id='convnet_mnist')
