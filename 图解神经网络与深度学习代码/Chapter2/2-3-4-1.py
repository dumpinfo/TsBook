# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列2深度的三维张量"
t3d=tf.constant(
        [
        [[1,2],[4,5],[6,7]],
        [[8,9],[10,11],[12,13]]
        ],tf.float32
        )
session=tf.Session()
#"改变形状为4行1列3深度的三维张量"
t1 = tf.reshape(t3d,[4,1,-1])
#"打印结果"
print(session.run(t1))