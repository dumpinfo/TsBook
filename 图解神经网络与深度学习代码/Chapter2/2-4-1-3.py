# -*- coding: utf-8 -*-
import tensorflow as tf
#"四维张量"
x=tf.constant(
        [
        #"第1个2行2列2深度的三维张量"
        [
        [[2,5],[4,3]],
        [[6,1],[1,2]]
        ],
        #"第2个2行2列2深度的三维张量"
        [
        [[3,9],[5,7]],
        [[7,5],[2,6]]
        ]
        ],tf.float32
        )
#"长度为2的一维张量"
y=tf.constant([10,20],tf.float32)
#"加法运算"
result=tf.add(x,y)
#"创建会话，打印结果"
session=tf.Session()
print(session.run(result))