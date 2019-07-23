# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"占位符"
x=tf.placeholder(tf.float32,[2,None],name='x')
# "3行2列矩阵"
w=tf.constant(
        [
        [1,2],
        [3,4],
        [5,6]
        ],tf.float32
        )
#"矩阵相乘"
y=tf.matmul(w,x)
#"创建会话"
session=tf.Session()
#"令x为2行2列的矩阵"
result1=session.run(y,feed_dict={x:np.array([[2,1],[1,2]],np.float32)})
print(result1)
#"令x为2行1列的矩阵"
result2=session.run(y,feed_dict={x:np.array([[-1],[2]],np.float32)})
print(result2)