# -*- coding: utf-8 -*-
import tensorflow as tf
#"2行3列的二维张量"
value1=tf.constant(
        [
         [1,2,3],
         [4,5,6]
        ],tf.float32)
#"2行1列的二维张量"
value2=tf.constant([
        [10],
        [20]
        ],tf.float32)
#"加法运算"
result=tf.add(value1,value2)
#"创建会话"
session=tf.Session()
#""打印结果""
print(session.run(result))