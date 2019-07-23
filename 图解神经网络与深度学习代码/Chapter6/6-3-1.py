# -*- coding: utf-8 -*-
import tensorflow as tf
#"输出层的值"
logits=tf.constant([[ -8.286214,0.64386976,9.21543,-0.07865417,5.6011457,
                     6.145635,-10.207598,7.5121603,7.7261553,9.431863]],
                     tf.float32)
#"人工分类的标签"
labels=tf.constant([[1,0,0,0,0,0,0,0,0,0]],tf.float32)
#"sigmod交叉熵"
entroy=tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels)
#"损失值"
loss=tf.reduce_sum(entroy)
#"打印损失值"
session=tf.Session()
print(session.run(loss))