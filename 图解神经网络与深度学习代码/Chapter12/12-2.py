# -*- coding: utf-8 -*-
import tensorflow as tf
#"初始化x的值"
x=tf.Variable(tf.constant([
                           [
                           [[8],[2],[9],[3]],
                           [[4],[6],[7],[10]],
                           [[20],[13],[1],[5]],
                           [[12],[18],[19],[14]]
                           ]
                           ],tf.float32),dtype=tf.float32)
#"2x2掩码，步长为2x2的最大化池化操作"
x_maxPool=tf.nn.max_pool(x,(1,2,2,1),(1,2,2,1),'VALID')
#"对以上的最大化池化结果计算其平方和"
F=tf.reduce_sum(tf.square(x_maxPool))
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
#"梯度下降法"
opti=tf.train.GradientDescentOptimizer(0.5).minimize(F)
#"打印前2次的结果"
for i in range(2):
    session.run(opti)
    print(session.run(x))