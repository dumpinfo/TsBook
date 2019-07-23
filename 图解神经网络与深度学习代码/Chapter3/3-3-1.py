# -*- coding: utf-8 -*-
import tensorflow as tf
#"初始化变量x的值"
x=tf.Variable(tf.constant([[4],[3]],tf.float32),dtype=tf.float32)
w=tf.constant([[1,2]],tf.float32)
y=tf.reduce_sum(tf.matmul(w,tf.square(x)))
#"Adagrad的梯度下降法"
opti=tf.train.AdagradOptimizer(0.25,0.1).minimize(y)
session=tf.Session()
init=tf.global_variables_initializer()
session.run(init)
#"打印前三次的迭代结果"
for i in range(3):
    session.run(opti)
    print(session.run(x))