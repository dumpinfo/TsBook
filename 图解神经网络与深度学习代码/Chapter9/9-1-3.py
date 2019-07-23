# -*- coding: utf-8 -*-
import tensorflow as tf
value3d=tf.constant(
        [
        #"第1个3行3列2深度的三维张量"
        [
          [[2,5],[3,3],[8,2]],
          [[6,1],[1,2],[5,4]],
          [[7,9],[2,-3],[-1,3]]
         ],
        #"第2个3行3列2深度的三维张量"
        [
        [[1,4],[9,3],[1,1]],
        [[1,1],[1,2],[3,3]],
        [[2,1],[3,6],[4,2]]
        ]
        ],tf.float32
        )
#"2行2列的池化掩码,在行方向上的移动步长为2,在列方向上的移动步长为2"
valued3d_maxPool=tf.nn.max_pool(value3d,(1,2,2,1),[1,2,2,1],'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(valued3d_maxPool))