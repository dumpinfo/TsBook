# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
t=tf.constant([[1,3],[2,0]],tf.float32)
#"sigmod激活"
result=tf.nn.sigmoid(t)
#"创建会话"
session=tf.Session()