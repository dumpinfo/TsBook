# -*- coding: utf-8 -*-
import tensorflow as tf
#"初始化变量2个变量，形状与model.cpkt文件中变量是必须相等的"
v1=tf.Variable([11,12,13],dtype=tf.float32,name='v1')
v2=tf.Variable([15,16],dtype=tf.float32,name='v2')
#"声明一个tf.train.Saver类"
saver =tf.train.Saver()
with tf.Session() as sess:
    #"加载model.ckpt文件"
    saver.restore(sess,'./model.ckpt')
    #"打印两个变量的值"
    print(sess.run(v1))
    print(sess.run(v2))
sess.close()