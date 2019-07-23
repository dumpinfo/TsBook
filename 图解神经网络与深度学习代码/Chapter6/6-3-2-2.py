# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入张量"
x=tf.constant([[1,2,1],
               [2,2,2]],tf.float32)
#"分别对每一行（沿"1"方向）进行softmax处理"
s=tf.nn.softmax(x,1)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(s))