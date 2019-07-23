# -*- coding: utf-8 -*-
import tensorflow as tf
#"一维张量"
t=tf.constant([1,2,3],tf.float32)
#"创建会话"
session=tf.Session()
#"张量转换为ndarray"
array=session.run(t)
#"打印其数据结构类型及对应值"
print(type(array))
print(array)