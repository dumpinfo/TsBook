# -*- coding: utf-8 -*-
import tensorflow as tf
#"四维张量"
t4d=tf.constant(
        [
        #"第1个 高2宽3深度2的三维张量"
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]]
        ],
        #"第2个 高2宽3深度2的三维张量"
        [
        [[1,2],[3,6],[1,2]],
        [[3,1],[1,2],[2,1]]
        ]
        ],tf.float32
        )
#"转换为高为2的二维张量"
t2d=tf.reshape(t4d,[2,-1])
#t2d=tf.reshape(t4d,[-1,3*3*2])
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(t2d))