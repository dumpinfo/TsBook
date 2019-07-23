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
from libs.activations import lrelu
from libs.utils import corrupt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
#------------------------------

#10 定义Autoencoder自编码函数
def autoencoder(input_shape=[None, 784],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """定义一个去噪自编码函数.

    Parameters参数
    ----------
    input_shape 输入数据形状: list列表格式,可选
    n_filters 过滤器数目: list列表格式,可选
    filter_sizes 过滤器尺寸: list列表格式,可选

    Returns返回值
    -------
    x : Tensor张量，占位符，输入参数
    z : Tensor张量，内部参数，信息表达
    y : Tensor张量，基于输入数据的重构输出
    cost : Tensor张量，训练代价总数值

    Raises溢出，变量错误    
    """
    
    #10.1 神经网络输入数据
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

		#10.2 确保2-d二维数据转换为平方张量tensor。
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    #10.3 选择参数，应用于降噪自编码
    if corruption:
        current_input = corrupt(current_input)

    #10.4 设计encoder编码器
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    #10.5  保存信息表达数据
    z = current_input
    encoder.reverse()
    shapes.reverse()

    #10.6 设计decoder解码器，使用相同的权重参数
    for layer_i, shape in enumerate(shapes):
        W = encoder[layer_i]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    #10.7 重构神经网络
    y = current_input
    
    
    #10.8 测量像素间的差异，返回loss损失函数的结果数值
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    #
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


#20 mnist测试函数
def test_mnist():
    # 使用MNIST数据集，测试卷积自编码模型
    
    #20.1，读取 MNIST 数据
    mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    
    #20.2 设置自编码模型ae
    ae = autoencoder()
    print('\n#20.2,',ae)

    #20.3 设置优化函数
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    #20.4
    print('\n#20.4,设置Session变量，初始化所有graph图计算的所有变量')
    print('使用summary日志函数，保存graph图计算结构图')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rlog='/ailib/log_tmp'
    xsum= tf.summary.FileWriter(rlog, sess.graph) 
    
    #20.5，
    print('\n#20.5,开始训练，迭代次数n_epochs=10，每次训练批量batch_size = 100')
    batch_size = 100
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
    print('\n#20.6,重建案例图像')
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()



# %%
if __name__ == '__main__':
    test_mnist()
