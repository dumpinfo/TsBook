# -*- coding: utf-8 -*-
import tensorflow as tf
#"用字典类管理变量"
weights={
        'w1':tf.Variable([1,13,22],dtype=tf.float32,name='w1'),
        'w2':tf.Variable([31,32],dtype=tf.float32,name='w2')
        }
bias={
      'b1':tf.Variable([2,12],dtype=tf.float32,name='b1'),
      'b2':tf.Variable(23,dtype=tf.float32,name='b2')
      }
#"声明一个tf.train.Saver类"
saver=tf.train.Saver()
with tf.Session() as sess:
    #"加载 modelMul.ckpt 文件"
    saver.restore(sess,'./modelMul.ckpt')
    #"打印值"
    print(sess.run(weights['w1']))
    print(sess.run(weights['w2']))
    print(sess.run(bias['b1']))
    print(sess.run(bias['b2']))
sess.close()