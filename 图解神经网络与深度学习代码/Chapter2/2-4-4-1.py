# -*- coding: utf-8 -*-
import tensorflow as tf
#"长度为3的一维张量"
t1=tf.constant([1,2,3],tf.float32)
#"长度为3的一维张量"
t2=tf.constant([7,8,9],tf.float32)
#"在0方向上堆叠"
t=tf.stack([t1,t2],0)
session=tf.Session()
#"打印结果"
print(session.run(t))