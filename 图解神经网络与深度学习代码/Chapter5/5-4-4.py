# -*- coding: utf-8 -*-
import tensorflow as tf
#"变量"
x=tf.Variable(tf.constant([[2,1,3]],tf.float32))
w=tf.constant([[2],[3],[4]],tf.float32)
#"函数g"
g=tf.matmul(x,w)
#"函数f=leaky\_relu(g)"
f=tf.nn.leaky_relu(g,alpha=0.2)
#"牛顿梯度下降法"
opti=tf.train.GradientDescentOptimizer(0.5).minimize(f)
session=tf.Session()
session.run(tf.global_variables_initializer())
#"打印结果"
for i in range(3):
    session.run(opti)
    print('第%d次迭代值'%(i+1))
    print(session.run(x))