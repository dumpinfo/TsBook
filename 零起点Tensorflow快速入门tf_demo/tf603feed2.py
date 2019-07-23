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
a = tf.placeholder(tf.float32,name='ta')
b = tf.placeholder(tf.float32,name='tb')
c = tf.multiply(a, b,name='tc')
init = tf.global_variables_initializer()

#2
print('\n#2,session')
with tf.Session() as ss:
  #2.1  
  ss.run(init)
  
  #2.2
  xss=ss.run([c], feed_dict={a:[2.,3.,4.], b:[1.,2.,3.]})  
  print('#2.2,xss,',xss)
  
  #2.3
  print('\n#2.3,summary，rlog',rlog)
  xsum= tf.summary.FileWriter(rlog, ss.graph)  
