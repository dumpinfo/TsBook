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
a = tf.constant(3.0,name='ta')
b = tf.constant(2.0,name='tb')
c = tf.constant(5.0,name='tc')

#2
print('\n#2,set.model')
m1= tf.add(b, c,name='m1')
m2= tf.multiply(a, m1,name='m2')

#3
print('\n#3,session')
with tf.Session() as ss:
  #3.1  
  #ss.run(init)
  
  #3.2
  xss = ss.run([m2, m1])
  print('#3.2,xss,',xss)
  
  #3.3
  print('\n#3.3,summary，rlog',rlog)
  xsum= tf.summary.FileWriter(rlog, ss.graph)  
