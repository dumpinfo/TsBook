# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列2深度的三维张量"
x=tf.constant(
        [
        [[2,5],[3,4],[8,2]],
        [[6,1],[1,2],[5,4]]
        ],tf.float32
        )
#"创建会话"
session = tf.Session()
#"每一个(0,1)平面的转置"
r=tf.transpose(x,perm=[1,0,2])
print(session.run(r))