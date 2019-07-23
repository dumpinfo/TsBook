# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入张量I"
I=tf.constant(
        [
         [[3],[4],[1],[5],[6]]
                ],tf.float32
        )
#"卷积核"
K=tf.constant(
        [
        [[-1]],[[-2]],[[2]],[[1]]
                ],tf.float32
        )
#"same卷积"
I_conv1d_K=tf.nn.conv1d(I,K,1,'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(I_conv1d_K))