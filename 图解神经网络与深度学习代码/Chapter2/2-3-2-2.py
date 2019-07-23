# -*- coding: utf-8 -*-
import tensorflow as tf
#"3行4列的二维张量"
t2=tf.constant(
        [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]
        ],tf.float32
        )
#"从[0,1]位置开始,取高2宽2的区域"
t=tf.slice(t2,[0,1],[2,2])
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(t))