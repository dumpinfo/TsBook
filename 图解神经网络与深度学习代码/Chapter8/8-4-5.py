# -*- coding: utf-8 -*-
import tensorflow as tf
#"1个 3 行 2 列深度为 2 的张量"
x=tf.constant(
        [
        [
         [[2,5],[3,3],[8,2]],
         [[6,1],[1,2],[5,4]],
         [[7,9],[2,3],[-1,3]]
                ]
                ],tf.float32
        )
#" 3个 2 行 2 列深度为 2 的卷积核"
k=tf.constant(
        [
        [[[3,1,-3],[1,-1,7]],[[-2,2,-5],[2,7,3]]],
        [[[-1,3,1],[-3,-8,6]],[[4,6,8],[5,9,-5]]]
                ],tf.float32
        )
#
#"每一深度分别计算卷积"
x_depthwise_conv2d_k=tf.nn.depthwise_conv2d(x,k,[1,1,1,1],'VALID')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(x_depthwise_conv2d_k))