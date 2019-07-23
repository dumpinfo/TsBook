# -*- coding: utf-8 -*-
import tensorflow as tf
#"张量"
t=tf.constant(
        [
        [False,True,False],
        [False,False,True]
        ]
        ,tf.bool)
session=tf.Session()
#"bool型转换为数值型"
r=tf.cast(t,tf.float32)
print(session.run(r))