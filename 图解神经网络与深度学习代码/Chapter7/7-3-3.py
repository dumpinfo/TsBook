# -*- coding: utf-8 -*-
import tensorflow as tf
#"3个长度是3深度是3的张量"
x=tf.constant(
        [
        [[2,5,2],[6,1,-1],[7,9,-5]], #"第1个长度是3深度是3的张量"
        [[1,3,2],[5,2,-2],[8,4,3]],  #"第2个长度是3深度是3的张量"
        [[4,5,-1],[1,9,5],[2,7,0]]   #"第3个长度是3深度是3的张量"
                ],tf.float32
        )
#"2个长度是2深度是3的卷积核"
k=tf.constant(
        [
        [[-1,1],[5,3],[4,7]],[[2,-2],[1,-1],[6,9]]
                ],tf.float32
        )
#"一维same卷积"
v_conv1d_k=tf.nn.conv1d(x,k,1,'SAME')
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(v_conv1d_k))