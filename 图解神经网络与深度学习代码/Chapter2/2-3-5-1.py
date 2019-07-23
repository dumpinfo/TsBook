# -*- coding: utf-8 -*-
import tensorflow as tf
#"一维张量"
t1d=tf.constant([3,4,1,5],tf.float32)
#"求和"
sum0=tf.reduce_sum(t1d)
#"打印结果"
session=tf.Session()
print(session.run(sum0))