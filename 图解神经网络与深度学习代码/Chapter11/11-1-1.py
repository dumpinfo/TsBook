# -*- coding: utf-8 -*-
import tensorflow as tf
#"卷积核"
kernel=tf.constant(
        [
        [[[3]],[[4]]],
        [[[5]],[[6]]]
        ],tf.float32
        )
#"某一函数针对 sigma 的导数"
out=tf.constant(
        [
        [
        [[-1],[1]],
        [[2],[-2]]
        ]
        ],tf.float32
        )
#"针对未知张量的导数的反向计算"
inputValue=tf.nn.conv2d_backprop_input((1,3,3,1),
                kernel,out,[1,1,1,1],'VALID')
#"创建会话"
session=tf.Session()
print(session.run(inputValue)