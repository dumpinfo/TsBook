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
from libs.utils import *
import matplotlib.pyplot as plt

#-----------------------
#1
print('\n#1,数据设置')
rlog='/ailib/log_tmp'
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


#2
print('\n#2,修改张量x形状参数shape')
# 张量 x当前形状spae[批量，高*宽]，需要重塑reshape为4-D tensor四维张量格式，以便兼容卷积graph图计算。
# -1是shape形状的是特殊值，表示任意大小1
x_tensor = tf.reshape(x, [-1, 28, 28, 1])

#3
print('\n#3,建立一个卷积层')
#权重矩阵（Weight matrix）是[height x width x input_channels x output_channels]
filter_size = 5
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

#4
print('\n#4,Bias偏差值是[output_channels]')
b_conv1 = bias_variable([n_filters_1])


#5
print('\n#5,建立一个图做卷积的第一层')
#每次迭代训练的幅度是 batch x height x width x channels
#我们使用2层和更多层的过滤层，替代池化层，以简化内部结构
h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=x_tensor,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)


#6
print('\n#6,和第一层一样，我们建立更深的一个图层')
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)

#7
print('\n#7,重塑reshape参数，以连接到全连接层')
# %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * n_filters_2])


#8
print('\n#8,建立全连接层')
n_fc = 1024
W_fc1 = weight_variable([7 * 7 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

#9
print('\n#9,添加dropout丢弃层，以减少过拟合，更加规范化')
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#10
print('\n#10,最后，加入softmax层，生成最终的预测数据')
# %% And finally our softmax layer:
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#11
print('\n#11,设置loss损失函数，eval评估函数、optimizer优化训练（training）函数')
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

#12
print('\n#12,设置准确度计算函数')
# %% Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#13
print('\n#13,设置Session变量，初始化所有graph图计算的所有变量')
print('使用summary日志函数，保存graph图计算结构图')
sess = tf.Session()
sess.run(tf.global_variables_initializer())
xsum= tf.summary.FileWriter(rlog, sess.graph) 

#14
print('\n#14,开始训练，迭代次数n_epochs=10，每次训练批量batch_size = 100')
batch_size = 100
n_epochs = 5
for epoch_i in range(n_epochs):
    #14.1 按batch_size批量训练摸
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.5})
    
    #14.2 计算每次迭代的相关参数
    xdat=sess.run(accuracy,feed_dict={x: mnist.validation.images,y: mnist.validation.labels,keep_prob: 1.0})
    print(epoch_i,'#',xdat)

#15
print('\n#15,使用test测试数据和训练好模型，计算相关参数')
xdat=sess.run(accuracy,feed_dict={x: mnist.test.images,y: mnist.test.labels,keep_prob: 1.0})
print(xdat)

#16
print('\n#16,查看和权重卷积函数的示意图')
W = sess.run(W_conv1)
plt.imshow(montage(W / np.max(W)), cmap='coolwarm')
