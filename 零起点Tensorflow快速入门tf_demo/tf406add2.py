#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import tensorflow as tf

#-----------------


#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
z = tf.add(x, y, name='z')


#2
print('\n#2,Session')
ss = tf.Session()
xsum= tf.summary.FileWriter(rlog, ss.graph)  

#3
print('\n#3,Session.run')
xss=ss.run(z, feed_dict={x:1, y:2.0})
print(xss)

#4
print('\n#4,Session.close')
ss.close()


