# -*- coding: utf-8 -*-
import tensorflow as tf
#"变量"
x=tf.Variable(tf.constant([[2,1,3]],tf.float32))
w=tf.constant([[2],[3],[4]],tf.float32)
#"函数g"
g=tf.matmul(x,w)
#"函数f=relu(g)"
f=tf.nn.relu(g)
#"计算f在(2,1,3)处的导数"
gradient=tf.gradients(f,[g,x])
session=tf.Session()
session.run(tf.global_variables_initializer())
#"打印结果"
print(session.run(gradient))