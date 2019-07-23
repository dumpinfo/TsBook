# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"第一步：已知数据"
xy=tf.placeholder(tf.float32,[None,2])
z=tf.placeholder(tf.float32,[None,1])
#"第二步：初始化变量"
w=tf.Variable(tf.constant([[1],[1]],tf.float32),dtype=tf.float32,name='w')
#"第三步：构造损失函数"
loss=tf.reduce_sum(tf.square(z-tf.matmul(xy,w)))
#"第四步：梯度下降，求解变量"
opti=tf.train.GradientDescentOptimizer(0.005).minimize(loss)
#"训练数据"
xy_train=np.array([[1,1],[2,1],[3,2],
                   [1,2],[4,5],[5,8]
                   ],np.float32)
z_train=np.array([[8],[12],[10],[14],[28],[10]],np.float32)
#"第五步：创建会话，训练模型"
session=tf.Session()
for i in range(500):
    session.run(opti,feed_dict={xy:xy_train,z:z_train})