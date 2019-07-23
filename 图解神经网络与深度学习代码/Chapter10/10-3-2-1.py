# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入的二维张量"
t=tf.constant(
        [
        [1,3,2,6],
        [7,5,4,9]
        ],tf.float32
        )
#"dropout处理"
r=tf.nn.dropout(t,0.5)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(r))