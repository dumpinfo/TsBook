# -*- coding: utf-8 -*-
import tensorflow as tf
t1=tf.constant(
        [
        [11,12,13],
        [14,15,16]
        ],
        tf.float32
        )
t2=tf.constant(
        [
        [4,5,6],
        [7,8,9]
        ],tf.float32
        )
session=tf.Session()
#"在方向0上堆叠"
t=tf.stack([t1,t2],0)
#"打印结果"
print(session.run(t))