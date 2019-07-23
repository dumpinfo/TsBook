# -*- coding: utf-8 -*-
import tensorflow as tf
#" 输入张量 5x5 "
I=tf.constant(
        [
        [
        [[2],[9],[11],[4],[8]],
        [[6],[12],[20],[16],[5]],
        [[1],[32],[13],[14],[10]],
        [[11],[20],[27],[40],[17]],
        [[9],[8],[11],[4],[1]]
        ]
        ],tf.float32
        )
# "卷积核 3x3"
Kernel=tf.constant(
        [
        [[[4]],[[8]],[[12]]],
        [[[5]],[[10]],[[15]]],
        [[[6]],[[12]],[[18]]]
        ],tf.float32
        )
#" 创建会话"
session=tf.Session()
# "输入张量与卷积核直接卷积"
result=tf.nn.conv2d(I,Kernel,[1,1,1,1],'SAME')
print('直接卷积的结果')
print(session.run(result))
#"卷积核可以分离为 3x1的垂直卷积核 和 1x3 水平卷积核"
kernel1=tf.constant(
        [
        [[[4]]],
        [[[5]]],
        [[[6]]]
        ],tf.float32
        )
kernel2=tf.constant(
        [
        [[[3]],[[2]],[[1]]]
        ],tf.float32
        )
#"将kernel2翻转180°"
rotate180_kernel2=tf.reverse(kernel2,axis=[1])
#"输入张量与分离的卷积核的卷积"
result1=tf.nn.conv2d(I,kernel1,[1,1,1,1],'SAME')
result12=tf.nn.conv2d(result1,rotate180_kernel2,[1,1,1,1],'SAME')
print('利用卷积核的分离性的卷积结果')
print(session.run(result12))