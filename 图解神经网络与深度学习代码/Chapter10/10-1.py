# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"输入张量"
input_tensor=tf.placeholder(tf.float32,[None,3,3,2])
#"3个 高2宽2深度2的卷积核"
kernel=tf.constant(
        [
        [ [ [-1,1,0],[1,-1,-1] ],[ [0,0,-1],[0,0,0] ] ],
        [ [ [0,0,0],[0,0,1] ], [ [1,-1,1],[-1,1,0] ]  ]
        ],tf.float32
        )
#"卷积"
conv2d=tf.nn.conv2d(input_tensor,kernel,(1,1,1,1),'SAME')
#"偏置"
bias=tf.constant([1,2,3],tf.float32)
conv2d_add_bias=tf.add(conv2d,bias)
#"激活函数"
active=tf.nn.relu(conv2d_add_bias)
#"pool操作"
active_maxPool=tf.nn.max_pool(active,(1,2,2,1),(1,1,1,1),'VALID')
#"拉伸"
shape=active_maxPool.get_shape ()
num=shape[1].value*shape[2].value*shape[3].value
flatten=tf.reshape(active_maxPool,[-1,num])
#flatten=tf.contrib.layers.flatten(active_maxPool)
#"打印结果"
session=tf.Session()
print(session.run(flatten,feed_dict={
        input_tensor:np.array([
                         #"第1个 3行3列2深度的三维张量"
                         [
                         [[2,5],[3,3],[8,2]],
                         [[6,1],[1,2],[5,4]],
                         [[7,9],[2,8],[1,3]]
                         ],
                         #"第2个 3行3列2深度的三维张量"
                         [
                         [[1,2],[3,6],[1,2]],
                         [[3,1],[1,2],[2,1]],
                         [[4,5],[2,7],[1,2]]
                         ],
                         #"第3个 3行3列2深度的三维张量"
                         [
                         [[2,3],[3,2],[1,2]],
                         [[4,1],[3,2],[1,2]],
                         [[1,0],[4,1],[4,3]]
                         ],
                         ],np.float32)}))