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
print('\n#1,set gg')
gg = tf.get_default_graph()

#2
print('\n#2')
op1=gg.get_operations()
print('op,',op1)
print('type(op),',type(op1))

#3
print('\n#3,in_dat')
in_dat = tf.constant(1.0)
op2=gg.get_operations()
print('in_dat,',in_dat)
print('type(in_dat),',type(in_dat))
print('\nop,',op2)
print('type(op),',type(op2))
print('\nop.node_def,',op2[0].node_def)

#4
print('\n#4,Session')
ss = tf.Session()
xdat=ss.run(in_dat)
print('xdat',xdat)


#5
print('\n#5,Session.close')
ss.close()


