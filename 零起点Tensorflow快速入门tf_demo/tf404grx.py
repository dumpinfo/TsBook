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
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
rlog='/ailib/log_tmp'

#2
print('\n#2,Session')
ss = tf.Session()


#3
print('\n#3,summary')
xsum= tf.summary.FileWriter(rlog, ss.graph)  
print('rlog',rlog)


#4
print('\n#4,Session.close')
ss.close()


