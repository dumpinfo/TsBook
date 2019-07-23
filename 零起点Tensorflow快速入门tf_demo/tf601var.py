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
cnt = tf.Variable(0, name="cnt")
a = tf.constant(1, name="a")

#2
print('\n#2,set.y')
y = tf.add(cnt, a)

#3
print('\n#4,set.y2')
y2 = tf.assign(cnt,y)

#4
print('\n#4,set.init')
init = tf.initialize_all_variables()

#5
print('\n#5,sess')
with tf.Session() as ss:
  #5.1  
  ss.run(init)
  #5.2
  xss=ss.run(cnt)
  print('x.cnt,',xss)
  #5.3
  for xc in range(3):
    ys2=ss.run(y2)
    #print('ys2,',ys2)
    xs2=ss.run(cnt)
    print('x2.cnt,',xs2)

  #5.4
  print('\n#5.4,summary，rlog',rlog)
  xsum= tf.summary.FileWriter(rlog, ss.graph)  

