# -*- coding: utf-8 -*-
import tensorflow as tf
#"卷积核"
kernel=tf.constant(
        [
        [[[3]],[[4]]],
        [[[5]],[[6]]]
        ],tf.float32
        )
# "某一函数针 F 对 sigma 的导数"
partial_sigma=tf.constant(
        [
        [
        [[-1],[1],[3]],
        [[2],[-2],[-4]],
        [[-3],[4],[1]]
        ]
        ],tf.float32
        )
# "针对未知张量导数的反向计算"
partial_x=tf.nn.conv2d_backprop_input((1,3,3,1),kernel,
            partial_sigma,[1,1,1,1],'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(partial_x))