# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"输入层"
x=tf.placeholder(tf.float32,(2,None))
#"第1层的权重矩阵"
w1=tf.constant(
        [[1,4,7],
        [2,6,8]],tf.float32
        )
#"第1层的偏置"
b1=tf.constant(
        [
        [-4],
        [2],
        [1]
        ],tf.float32
        )
#"计算第1层的线性组合"
l1=tf.matmul(w1,x,True)+b1
#"激活 2*x"
sigma1=2*l1
#"第2层的权重矩阵"
w2=tf.constant(
        [[2,3],
         [1,-2],
         [-1,1]
         ],tf.float32
        )
#"第2层的偏置"
b2=tf.constant(
        [[5],[-3]],tf.float32
        )
#"计算第1层的线性组合"
l2=tf.matmul(w2,sigma1,True)+b2
#"激活 2*x"
sigma2=2*l2
#"创建会话"
session=tf.Session()
#"令x=[[3],[5]]"
print(session.run(sigma2,{x:np.array([[3],[5]],np.float32)}))