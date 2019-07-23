# -*- coding: utf-8 -*-
import tensorflow as tf
#"首先将变量初始化:梯度下降的初始点"
x=tf.Variable(4.0,dtype=tf.float32)
#"函数"
y=tf.pow(x-1,2.0)
#"梯度下降,学习率为0.25"
opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
#"三次迭代"
for i in range(3):
    session.run(opti)
    #"打印每次迭代的值"
    print(session.run(x))