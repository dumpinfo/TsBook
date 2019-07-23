#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用CNN卷积神经网络

@from:
https://www.tensorflow.org/get_started/mnist/

'''



from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

#------------def.fun.xxx


def weight_variable(shape):
  #根据输入的shape形状数据，生成一个TF变量Variable参数，并进行初始化，作为权重变量
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  #根据输入的shape形状数据，生成一个TF变量Variable参数，并进行初始化，作为偏移变量
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  #返回一个步长为1的2d二维卷积层
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  #对于特征数据，进行2倍增幅下采样  
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def deepnn(x):
  """
  deepnn函数，构建一个DNN深度神经网络模型，用于MNIST数据集手写数字分类.

  输入参数：
      x，输入张量形状是(N_examples, 784)，其中784代表标准MNIST数据集单个图片的像素数目。N_examples,表示输入的样本数据数目，-1表示全部数据。
  返回值：
      元组格式tuple (y, keep_prob)，是一个shape形状张量(N_examples, 10)，其中的数值，表示分类为数字0-9的概率值。keep_prob，是一个占位符参数，表示dropout的概率参数。

  """
  #1
  
  
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  
  #2
  # 第一层卷积运算，根据灰度图，生成32组特征图。
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  
  #3
  # 池化层，对数据进行2倍增幅下采样  
  h_pool1 = max_pool_2x2(h_conv1)
  
  #4
  # 第二卷积层，32组特征图转换为64组
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  
  #5
  # 第二池化层
  h_pool2 = max_pool_2x2(h_conv2)

  #6  
  #全连接层1，通过2次下采样操作，
  #输入的28x28图像，转换为7x7x64特征映射图，映射1024个特征点
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  #7
  #Dropout层-控制模型的复杂度，防止features特征点互相干扰。
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  #8
  #1024个features特征点，映射到10个类，每类为一个数字
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  
  #9
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                    
  #10                    
  return y_conv, keep_prob



#
#------------main----------
#
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

#2
print('\n#2,构建模型')
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 使用deep net深度神经网络
y_conv, keep_prob = deepnn(x)

#3
print('\n#3,定义loss损失函数和optimizer优化函数')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#4
print('\n#4,Session')
#sess = tf.InteractiveSession()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#
xsum= tf.summary.FileWriter(rlog, sess.graph)  

#5
print('\n#5,Train，nstep=300')
nstep=300
for i in range(nstep):
  #5.a set step#n,dat,run
  batch = mnist.train.next_batch(50)
  
  #5.b print.info
  if i % 20 == 0:
    feed={x: batch[0], y_: batch[1], keep_prob: 1.0}  
    train_accuracy = sess.run(accuracy,feed_dict=feed)
    print('%d#, training accuracy %g' % (i, train_accuracy))
    
  #5.c  
  sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#6
print('\n#6,Test测试模型训练结果')
xdat=sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print('test accuracy,',xdat )

#7
print('\n#7,Session.close')
sess.close()