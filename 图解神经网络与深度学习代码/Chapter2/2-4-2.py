# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行2列的矩阵"
x=tf.constant(
        [[1,2],[3,4]]
        ,tf.float32
        )
#"2行1列的矩阵"
w=tf.constant([[-1],[-2]],tf.float32)
#"矩阵的乘法"
y=tf.matmul(x,w)
#"创建会话"
session=tf.Session()
#"打印矩阵相乘后的结果"
print(session.run(y))