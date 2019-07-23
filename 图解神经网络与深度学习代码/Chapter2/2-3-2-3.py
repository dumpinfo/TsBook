# -*- coding: utf-8 -*-
import tensorflow as tf
#"3行3列2深度的三维张量"
t3d=tf.constant(
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]],
        [[7,9],[2,-3],[-1,3]]
        ],tf.float32
        )
#"从[1,0,1]位置处,取高2宽2深度1的区域"
t=tf.slice(t3d,[1,0,1],[2,2,1])
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(t))