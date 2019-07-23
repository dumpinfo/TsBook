# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入形状为[1,4,4,1]的张量"
value2d = tf.constant(
        [
        #"第1个4行4列1深度的三维张量"
        [
        [[2],[3],[8],[-2]],
        [[6],[1],[5],[9]],
        [[7],[2],[-1],[8]],
        [[1],[4],[3],[5]]
                ]
                ],tf.float32
        )
#"最大池化操作"
value2d_maxPool=tf.nn.max_pool(value2d,(1,2,3,1),[1,1,1,1],'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(value2d_maxPool))