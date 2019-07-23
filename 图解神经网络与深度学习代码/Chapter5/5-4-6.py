# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入张量"
t=tf.constant([-2,0,1],tf.float32)
#"celu激活函数"
r=tf.nn.crelu(t)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(r))