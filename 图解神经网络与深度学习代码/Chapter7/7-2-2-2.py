# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
x=tf.constant([[1,2,3],[4,5,6]],tf.float32)
#"常数边界扩充，上侧补1行10，下侧补2行10，右侧补1列10"
r=tf.pad(x,[[1,2],[0,1]],mode='CONSTANT',constant_values=10)
#"创建会话"
session=tf.Session()
#"打印边界扩充后的结果"
print(session.run(r))