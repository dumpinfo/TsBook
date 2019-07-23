# -*- coding: utf-8 -*-
import tensorflow as tf
v=tf.one_hot([9,2,7,3,0,4,8,6,1,3,4,8,6,1], depth=10,axis=1,dtype=tf.float32)
session=tf.Session()
print(session.run(v))