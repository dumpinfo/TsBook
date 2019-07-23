# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"占位符"
x=tf.placeholder(tf.float32,[None,2])
keep_prob=tf.placeholder(tf.float32)
#"输入层到隐含层的权重矩阵"
w1=tf.constant([
        [1,3,5],
        [2,4,6]
        ],tf.float32)
#"隐含层的值"
h1=tf.matmul(x,w1)
#"dropout层"
h1_dropout=tf.nn.dropout(h1,keep_prob)
#"dropout层到输出层的权重矩阵"
w2=tf.constant(
        [
        [8,3],
        [7,2],
        [6,1]
        ],tf.float32
        )
#"输出层的值"
o=tf.matmul(h1_dropout,w2)
x_input=np.array([[2,3],[1,4]],np.float32)
#"创建会话"
session=tf.Session()
h1_arr,h1_dropout_arr,o_arr=s=session.run(
        [h1,h1_dropout,o],feed_dict={x:x_input,keep_prob:0.5})
#"打印结果"
print('隐含层的值:')
print(h1_arr)
print("'dropout层的值:'")
print(h1_dropout_arr)
print('输出层的值:')
print(o_arr)