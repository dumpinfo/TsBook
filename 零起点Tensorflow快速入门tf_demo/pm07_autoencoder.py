#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

pkmital案例合集

@from:
pkmital案例合集网址：
https://github.com/pkmital/tensorflow_tutorials

'''

import tensorflow as tf
import numpy as np
import math
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt

#--------------------
#10 定义Autoencoder自编码函数
def autoencoder(dimensions=[784, 512, 256, 64]):
    """ 定义一个深自编码函数.

    Parameters参数
    ----------
    dimensions 维度: list列表合适, optional可选
    		每层神经元个数的编码。

    Returns返回值
    -------
    x : Tensor张量，占位符，输入参数
    z : Tensor张量，内部参数，信息表达
    y : Tensor张量，基于输入数据的重构输出
    cost : Tensor张量，训练代价总值
    """
    #10.1 神经网络输入数据
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x

    #10.2 设计encoder编码器
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    #10.3  信息表达
    z = current_input
    encoder.reverse()

    #10.4 设计decoder解码器，使用相同的权重参数
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    #10.5 重构神经网络
    y = current_input

    #10.6 测量像素间的差异，返回loss损失函数的结果数值
    cost = tf.reduce_sum(tf.square(y - x))
    #
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

#20 mnist测试函数
def test_mnist():
    #使用MNIST数据集，测试自编码模型

    #20.1 
    # 读取 MNIST 数据
    mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    
    #20.2 设置自编码模型ae
    ae = autoencoder(dimensions=[784, 256, 64])
    print('\n#20.2,',ae)

    #20.3 设置优化函数
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    #20.4
    print('\n#20.4,设置Session变量，初始化所有graph图计算的所有变量')
    print('使用summary日志函数，保存graph图计算结构图')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rlog='/ailib/log_tmp'
    xsum= tf.summary.FileWriter(rlog, sess.graph) 

    
    #20.5
    # Fit all training data
    print('\n#20.5,开始训练，迭代次数n_epochs=10，每次训练批量batch_size = 50')
    batch_size = 50
    n_epochs = 10
    for epoch_i in range(n_epochs):
    	  #20.5a 按batch_size批量训练摸
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
            	
        #20.5b 计算每次迭代的相关参数    	
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train}))

    # 20.6
    print('\n#20.6, 重建案例图像')
    n_examples = 15
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    fig.show()
    plt.draw()  



# %%
if __name__ == '__main__':
    test_mnist()
