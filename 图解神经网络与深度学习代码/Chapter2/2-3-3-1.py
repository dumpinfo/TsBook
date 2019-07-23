# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列的二维张量"
x=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ],tf.float32
        )
#"创建会话"
session = tf.Session()
#"转置"
r=tf.transpose(x,perm=[1,0])
print(session.run(r))