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
#" 1个 2 行 2 列深度为 2 的卷积核"
k=tf.constant(
        [
        [[[3],[1]],[[-2],[2]]],
        [[[-1],[-3]],[[4],[5]]]
                ],tf.float32
        )
#"每一深度分别计算卷积,然后求和"
x_conv2d_k=tf.nn.conv2d(x,k,[1,1,1,1],'VALID')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(x_conv2d_k))