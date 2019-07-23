# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
t1=tf.constant(
        [
        [1,5,7],
        [2,3,8]
        ],tf.float32
        )
#"二维张量"
t2=tf.constant(
        [
        [2,5,6],
        [7,1,8]
        ],tf.float32
        )
#"两个张量的对比"
r=tf.equal(t1,t2)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(r))