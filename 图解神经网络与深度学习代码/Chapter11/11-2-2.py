# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入张量"
x=tf.constant(
        [
        [
        [[1],[2],[3]],
        [[4],[5],[6]],
        [[7],[8],[9]]
        ]
        ],
        tf.float32
        )
# "某一函数 F 对 sigma 的导数"
partial_sigma=tf.constant(
        [
        [
        [[-1],[-2],[1]],
        [[-3],[-4],[2]],
        [[-2],[1],[3]]
        ]
        ],tf.float32
        )
# "某一函数 F 对 卷积核 k 的导数"
partial_sigma_k=tf.nn.conv2d_backprop_filter(x,(2,2,1,1),
                partial_sigma,[1,1,1,1],'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(partial_sigma_k))