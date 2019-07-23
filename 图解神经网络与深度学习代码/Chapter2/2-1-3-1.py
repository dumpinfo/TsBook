# -*- coding: utf-8 -*-
import tensorflow as tf
#"张量"
t=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ]
        ,tf.float32)
session=tf.Session()
#"张量的形状"
s=tf.shape(t)
print('张量的形状:',session.run(s))