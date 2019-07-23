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
a = tf.Variable(1.0, name='ta')
b = tf.constant(1.0, name='tb')
c = tf.multiply(a, b, name='tc')

#2
print('\n#2,Session')
ss = tf.Session()

#3
print('\n#3,summary,rlog',rlog)
xsum= tf.summary.FileWriter(rlog, ss.graph)  

#4
print('\n#4,Session init')
init=tf.global_variables_initializer()
ss.run(init)


#5
print('\n#5,Session.run')
xss=ss.run(c)
print('xss,',xss)


#6
print('\n#6,Session.close')
ss.close()
