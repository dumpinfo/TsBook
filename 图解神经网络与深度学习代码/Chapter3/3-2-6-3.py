# -*- coding: utf-8 -*-
import tensorflow as tf
#"梯度下降的初始点"
x1=tf.Variable(-4.0,dtype=tf.float32)
x2=tf.Variable(4.0,dtype=tf.float32)
#"函数"
y=tf.square(x1)+tf.square(x2)
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
#"梯度下降,设置步长为0.25"
opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)
#"2次迭代"
for i in range(2):
    session.run(opti)
    #"打印每次迭代的值"
    print((session.run(x1),session.run(x2)))