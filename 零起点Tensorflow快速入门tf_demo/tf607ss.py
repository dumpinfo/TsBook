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
a = tf.Variable(3, tf.int16,name='ta')
b = tf.Variable(3, tf.int16,name='tb')

#2
print('\n#2,set.model')
m1= tf.add(a, b,name='m1')

#3
print('\n#3,set.init')
init = tf.global_variables_initializer()

#4
print('\n#4,session')
with tf.Session() as ss:
  #4.1  
  ss.run(init)
  
  #4.2
  xss = ss.run(m1)
  print('#4.2,xss,',xss)
  
  #4.3
  print('\n#4.3,summary，rlog',rlog)
  xsum= tf.summary.FileWriter(rlog, ss.graph)  
