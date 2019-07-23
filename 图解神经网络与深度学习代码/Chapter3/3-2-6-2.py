# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
#"首先将变量初始化:梯度下降的初始点"
x=tf.Variable(15.0,dtype=tf.float32)
#"函数"
y=tf.pow(x-1,2.0)
#"梯度下降,设置学习率为0.25"
opti=tf.train.GradientDescentOptimizer(0.05).minimize(y)
#"画曲线"
value=np.arange(-15,17,0.01)
y_value=np.power(value-1,2.0)
plt.plot(value,y_value)
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
#"三次迭代"
for i in range(100):
    session.run(opti)
    if(i%10==0):
        v=session.run(x)
        plt.plot(v,math.pow(v-1,2.0),'go')
        print('第 %d 次的 x 的迭代值: %f'%(i+1,v))
plt.show()