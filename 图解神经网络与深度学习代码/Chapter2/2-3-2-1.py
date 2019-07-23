# -*- coding: utf-8 -*-
import tensorflow as tf
#"长度为5的一维张量"
t1=tf.constant([1,2,3,4,5],tf.float32)
#"从t1的第1个位置开始,取长度为3的区域"
t=tf.slice(t1,[1],[3])
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(t))