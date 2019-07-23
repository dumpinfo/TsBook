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
hello = tf.constant('Hello, TensorFlow!') 
ss = tf.Session() 
print(ss.run(hello))

#2
print('\n#2')
a = tf.constant(2) 
b = tf.constant(3) 
with tf.Session() as ss:     
    print('a=2, b=3')
    print('a+b=',ss.run(a+b))
    print('a*b=',ss.run(a*b))
    