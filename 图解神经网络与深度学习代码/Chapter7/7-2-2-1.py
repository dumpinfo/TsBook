# -*- coding: utf-8 -*-
import tensorflow as tf
#"一维张量"
x=tf.constant([2,1],tf.float32)
#"常数边界扩充，上侧补1个0，下侧补2个0"
r=tf.pad(x,[[1,2]],mode='CONSTANT')
#"创建会话"
session=tf.Session()
#"打印边界扩充后的结果"
print(session.run(r))