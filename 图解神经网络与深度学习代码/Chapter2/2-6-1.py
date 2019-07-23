# -*- coding: utf-8 -*-
import tensorflow as tf
#"创建1个 Variable 对象"
v=tf.Variable(tf.constant([2,3],tf.float32))
#"创建会话"
session=tf.Session()
#"Variable 对象初始化"
session.run(tf.global_variables_initializer())
#"打印值"
print('v初始化的值')
print(session.run(v))
#"利用成员函数 assign\_add 改变本身的值"
session.run(v.assign_add([10,20]))
print('v 的当前值')
print(session.run(v))