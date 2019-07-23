# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"输入层:每一个输入按行存储"
x=tf.placeholder(tf.float32,(None,2))
#"第1层的权重矩阵"
w1=tf.constant(
        [
        [1,4,7],
        [2,6,8],
        ],tf.float32
        )
#"第1层的偏置:"
b1=tf.constant([1,4,7],tf.float32)
#b1=tf.constant([[1,4,7]],tf.float32)#"以上b1可以这样写"
#"计算第一层的线性组合"
l1=tf.matmul(x,w1)+b1
#"激活 2*x"
sigma1=2*l1
#"第2层的权重矩阵"
w2=tf.constant(
        [[2,3],
         [1,-2],
         [-1,1]],tf.float32
        )
#"第1层的偏置"
b2=tf.constant(
        [5,-3],tf.float32
        )
#"计算第2层的线性组合"
l2=tf.matmul(sigma1,w2)+b2
#"激活 2*x"
sigma2=2*l2
#"创建会话"
session=tf.Session()
#"4个输入"
print(session.run(sigma2,{x:np.array([
                                     [10,11],
                                     [20,21],
                                     [30,31],
                                     [40,41]],np.float32)}))