# -*- coding: utf-8 -*-
import tensorflow as tf
#"张量"
t=tf.constant(
        [
        [0,2,0],
        [0,0,1]
        ]
        ,tf.float32)
session=tf.Session()
#"数值型转换为bool类型"
r=tf.cast(t,tf.bool)
print(session.run(r))