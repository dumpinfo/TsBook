# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
t=tf.constant([[1,2,3],[4,5,6]],tf.float32)
#"水平镜像"
rh=tf.reverse(t,axis=[0])
#"垂直镜像"
rv=tf.reverse(t,axis=[1])
#"逆时针翻转180°：即先水平镜像后垂直镜像(或者先垂直镜像后水平方向)"
r=tf.reverse(t,axis=[0,1])
#"创建会话"
session=tf.Session()
#"打印结果"
print('水平镜像的结果')
print(session.run(rh))
print('垂直镜像的结果')
print(session.run(rv))
print('逆时针翻转180的结果')
print(session.run(r))