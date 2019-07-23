# -*- coding: utf-8 -*-
import tensorflow as tf
x=tf.constant([1,10,23,15],tf.float32)
#"计算均值和方差"
mean,variance=tf.nn.moments(x,[0])
#"BatchNormalize"
r=tf.nn.batch_normalization(x,mean,variance,0,1,1e-8)
session=tf.Session()
print(session.run(r))