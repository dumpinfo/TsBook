# -*- coding: utf-8 -*-
import tensorflow as tf
#"用字典类管理变量"
weights={
        'w1':tf.Variable([11,12,13],dtype=tf.float32,name='w1'),
        'w2':tf.Variable([21,22],dtype=tf.float32,name='w2')
        }
bias={
      'b1':tf.Variable([101,102],dtype=tf.float32,name='b1'),
      'b2':tf.Variable(2,dtype=tf.float32,name='b2')
      }
#"创建会话"
session=tf.Session()
#"声明一个tf.train.Saver类"
saver=tf.train.Saver()
with tf.Session() as sess:
    #"变量初始化"
    sess.run(tf.global_variables_initializer())
    #"将变量保存在当前文件夹下的 modelMul.ckpt 文件中"
    saver.save(sess,'./modelMul.ckpt')