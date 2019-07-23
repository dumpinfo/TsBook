# -*- coding: utf-8 -*-
import tensorflow as tf
inputValue=tf.constant(
        [
        #"一个深度是2 且3行3列的张量"
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]],
        [[7,9],[2,3],[-1,3]]
        ]
        ],tf.float32
        )

#"3个深度是2且为2行2 列的卷积核"
kernels=tf.constant(
        [
        [[[3,1,-3],[1,-1,7]],[[-2,2,-5],[2,7,3]]],
        [[[-1,3,1],[-3,-8,6]],[[4,6,8],[5,9,-5]]]
        ],tf.float32
        )
#"valid卷积"
validResult=tf.nn.conv2d(inputValue,kernels,[1,1,1,1],'VALID')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(validResult))