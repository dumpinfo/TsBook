# -*- coding: utf-8 -*-
import tensorflow as tf
#"第1个Variable，初始化为一个长度为3的一维张量"
v1=tf.Variable(tf.constant([1,2,3],tf.float32),dtype=tf.float32,name='v1')
#"第2个Variable，初始化为一个长度为2的一维张量"
v2=tf.Variable(tf.constant([4,5],tf.float32),dtype=tf.float32,name='v2')
#"声明一个tf.train.Saver对象"
saver =tf.train.Saver()
#"创建会话"
session=tf.Session()
#"初始化变量"
session.run(tf.global_variables_initializer())
#"将变量 v1 和 v2 保存到当前文件夹下的model.ckpt文件中"
save_path=saver.save(session,'./model.ckpt')
session.close()