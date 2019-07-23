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
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

#-----------------------

#1
print('\n#1,数据设置')
rlog='/ailib/log_tmp'

#读取经典的MNIST数据集
#使用one-hot独热码，每个稀疏向量只有标签类值是1，其他类是0
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)

#2
print('\n#2,数据查看')
# 访问MNIST数据集，包括：
# 'train'训练, 'test'测试, 和 'validation'确认部分
#可以访问：images图像，labels标签，和数目
print('train,',mnist.train.num_examples)
print('test,',mnist.test.num_examples)
print('validation,',mnist.validation.num_examples)

#3
print('\n#3,image图像数据形状shape')
# 图像存储为： n个特征tensor张量（n维数组）
#labels标签存储为：n个one-hot独热码标签，
print('images.shape,',mnist.train.images.shape)
print('labels.shape,',mnist.train.labels.shape)

#4
print('\n#4,image图像数值范围是从0到1的灰度图')
print(np.min(mnist.train.images), np.max(mnist.train.images))

#5
print('\n#5,可视化的任一个图像（28x28像素）')
plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')

#6
print('\n#6,创建一个使用TF graph图计算的输入图像数据的容器：')
#第一个维度是None，表示使用全部收入数据，迭代时我们可以使用mini-batches小批次数据。
#第二个维度是输入的image图像的特征维度，案例当中，n_input = 784
n_input = 784
n_output = 10
net_input = tf.placeholder(tf.float32, [None, n_input])


#7
print('\n#7,使用最简单的线性回归模型：Y = W * X + B')
W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
net_output = tf.nn.softmax(tf.matmul(net_input, W) + b)

#8
print('\n#8,设置占位符')
y_true = tf.placeholder(tf.float32, [None, 10])

#9
print('\n#9,设置loss损失函数')
cross_entropy = -tf.reduce_sum(y_true * tf.log(net_output))


#10
print('\n#10,评估预测的正确率correct_prediction')
# 比较预测结果net_output与真实数据y_true，计算预测的正确率
# 预测结果是one-hot独热码，需要使用argmax函数转换
correct_prediction = tf.equal(
    tf.argmax(net_output, 1), tf.argmax(y_true, 1))

#11
print('\n#11,设置accuracy准确度计算函数')
# 计算准确度的平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#12
print('\n#12,使用梯度下降优化函数')
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#13
print('\n#13,设置Session变量，初始化所有graph图计算的所有变量')
print('使用summary日志函数，保存graph图计算结构图')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
xsum= tf.summary.FileWriter(rlog, sess.graph) 

#14
print('\n#14,开始训练，迭代次数n_epochs=10，每次训练批量batch_size = 100')
batch_size = 100
n_epochs = 10
for epoch_i in range(n_epochs):
    #14.1 按batch_size批量训练摸
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            net_input: batch_xs,
            y_true: batch_ys
        })
    
    #14.2 计算每次迭代的相关参数
    xdat=sess.run(accuracy,feed_dict={net_input: mnist.validation.images,y_true: mnist.validation.labels})
    print(epoch_i,'#',xdat)

#15
print('\n#15,使用test测试数据和训练好模型，计算相关参数')
xdat=sess.run(accuracy,feed_dict={net_input: mnist.test.images,y_true: mnist.test.labels})    
print(xdat)
