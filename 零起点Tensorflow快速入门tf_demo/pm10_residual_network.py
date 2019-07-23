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
from libs.connections import conv2d, linear
from collections import namedtuple
from math import sqrt
import tensorflow.examples.tutorials.mnist.input_data as input_data

#-----------------------
#10 定义深度残差神经网络函数
def residual_network(x, n_outputs,
                     activation=tf.nn.relu):
    """构建深度残差神经网络

    Parameters参数
    ----------
    x : Placeholder张量，占位符，输入参数
    n_outputs 输出数目: TYPE类型，最终softmax输出结果的数目
    activation 激活函数: Attribute属性，可选，每次卷积后使用的非线性函数

    Returns返回值
    -------
    net 网络: Tensor张量

    Raises溢出，
    ------
    ValueError，变量错误，
    		如果一个2D Tenso二维张量输入数据，张量必须是方形结构
    		或无法转换为其他4D Tensor张量的数据
    """
    
    #10.1 设置神经网络图层数据
    LayerBlock = namedtuple(
        'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
    blocks = [LayerBlock(3, 128, 32),
              LayerBlock(3, 256, 64),
              LayerBlock(3, 512, 128),
              LayerBlock(3, 1024, 256)]
    
    #10.2 根据输入数据形状shape，把2D Tenso二维张量输入数据，转换为方形结构
    input_shape = x.get_shape().as_list()
    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        if ndim * ndim != input_shape[1]:
            raise ValueError('input_shape should be square')
        x = tf.reshape(x, [-1, ndim, ndim, 1])

    #10.3 第一卷积函数，扩展到64通道，并且降低采样率
    net = conv2d(x, 64, k_h=7, k_w=7,
                 name='conv1',
                 activation=activation)

    #10.4 最大化池化函数，并且降低采样率
    net = tf.nn.max_pool(
        net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    #10.5 设置残差网络第一区块
    net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1,
                 stride_h=1, stride_w=1, padding='VALID', name='conv2')

    #10.6 遍历所有的区块block
    for block_i, block in enumerate(blocks):
        for repeat_i in range(block.num_repeats):

            name = 'block_%d/repeat_%d' % (block_i, repeat_i)
            conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_in')

            conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
                          padding='SAME', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_bottleneck')

            conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
                          padding='VALID', stride_h=1, stride_w=1,
                          activation=activation,
                          name=name + '/conv_out')

            net = conv + net
        try:
            #10.7 提示下一组区块尺寸
            next_block = blocks[block_i + 1]
            net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                         padding='SAME', stride_h=1, stride_w=1, bias=False,
                         name='block_%d/conv_upscale' % block_i)
        except IndexError:
            pass

    #10.8 平均化池化函数
    net = tf.nn.avg_pool(net,
                         ksize=[1, net.get_shape().as_list()[1],
                                net.get_shape().as_list()[2], 1],
                         strides=[1, 1, 1, 1], padding='VALID')
    #10.9 重塑网络形状reshape                     
    net = tf.reshape(
        net,
        [-1, net.get_shape().as_list()[1] *
         net.get_shape().as_list()[2] *
         net.get_shape().as_list()[3]])
         
		#10.10 对网络进行线性化运算
    net = linear(net, n_outputs, activation=tf.nn.softmax)

    #10.12 返回网络模型 
    return net

#20 mnist测试函数
def test_mnist():
		#使用MNIST数据集，测试残差网络
    
		#20.1，读取 MNIST 数据
    mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    
    #20.2，设置残差网络
    y_pred = residual_network(x, 10)
    print('\n#20.2,残差网络')
    print(y_pred)

    #20.3a 设置loss损失，eval评估、训练、优化函数
    cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    #20.3b 设置准确度评估函数
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    #20.4
    print('\n#20.4,设置Session变量，初始化所有graph图计算的所有变量')
    print('使用summary日志函数，保存graph图计算结构图')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rlog='/ailib/log_tmp'
    xsum= tf.summary.FileWriter(rlog, sess.graph) 
    
    #20.5，
    print('\n#20.5,开始训练，迭代次数n_epochs=5，每次训练批量batch_size = 50')
    batch_size = 50
    n_epochs = 5
    for epoch_i in range(n_epochs):
        #20.5a 按batch_size批量训练模型
        train_accuracy = 0
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            train_accuracy += sess.run([optimizer, accuracy], feed_dict={
                x: batch_xs, y: batch_ys})[1]
        train_accuracy /= (mnist.train.num_examples // batch_size)

        
        #20.5b 按batch_size批量验证模型
        valid_accuracy = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, batch_ys = mnist.validation.next_batch(batch_size)
            valid_accuracy += sess.run(accuracy,
                                       feed_dict={
                                           x: batch_xs,
                                           y: batch_ys
                                       })
        #20.5c 计算每次迭代的相关参数    	
        valid_accuracy /= (mnist.validation.num_examples // batch_size)
        print('epoch:', epoch_i, ', train:',
              train_accuracy, ', valid:', valid_accuracy)



if __name__ == '__main__':
    test_mnist()
