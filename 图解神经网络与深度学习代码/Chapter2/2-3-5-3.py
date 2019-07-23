# -*- coding: utf-8 -*-
import tensorflow as tf
#"三维张量"
value3d=tf.constant(
        [
        [[2,5],[3,3],[8,2]],
        [[6,1],[1,2],[5,4]],
        [[7,9],[2,-3],[-1,3]]
        ],tf.float32
        )
#"创建会话"
session=tf.Session()
#"计算沿 0 方向上的和"
sum0=tf.reduce_sum(value3d,axis=0)
print(session.run(sum0))