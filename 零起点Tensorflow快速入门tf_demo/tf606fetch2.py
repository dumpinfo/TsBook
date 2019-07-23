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
a_dat=[[1., 2., 3.], [4., 5., 6.]]
b_dat=[[2., 3., 4.], [3., 2., 1.]]
c_dat=[[3., 2., 1.], [1., 3., 3.]]
a = tf.constant(a_dat,name='ta')
b = tf.constant(b_dat,name='tb')
c = tf.constant(c_dat,name='tc')
  

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
  print('#3.2,xss,\n',xss)
  
  #3.3
  print('\n#3.3,summary，rlog',rlog)
  xsum= tf.summary.FileWriter(rlog, ss.graph)  
