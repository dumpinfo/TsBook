# -*- coding: utf-8 -*-
import tensorflow as tf
#"初始化变量，即 t=1 时的值"
x=tf.Variable(initial_value=5,dtype=tf.float32,trainable=False,name='v')
#"创建计算移动平均的对象"
exp_moving_avg=tf.train.ExponentialMovingAverage(0.7)
update_moving_avg=exp_moving_avg.apply([x])
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
for i in range(4):
    #"打印指数移动平均值"
    session.run(update_moving_avg)
    print('第{}次的移动平均值:'.format(i+1))
    print(session.run(exp_moving_avg.average(x)))
    session.run(x.assign_add(5))