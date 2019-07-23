# -*- coding: utf-8 -*-
import tensorflow as tf
#"假设\_y为全连接网络的输出(输出层有3个神经元)"
_y=tf.constant([[0,2,-3],[4,-5,6]],tf.float32)
#"人工分类结果"
y=tf.constant([[1,0,0],[0,0,1]],tf.float32)
#"softmax熵"
entroy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=_y,labels=y)
#"loss"
loss=tf.reduce_sum(entroy)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(loss))