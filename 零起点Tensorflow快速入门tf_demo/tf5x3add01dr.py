#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import tensorflow as tf
import numpy as np
import pandas as pd
#-----------------

#1
print('\n#1，set#1')
x = tf.placeholder("float",shape=[None,1])
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

xsum = np.empty(shape=[1,1])
ysum = np.empty(shape=[1,1])
df=pd.DataFrame(columns=['y_sr','y_pred'])
#2
print('\n#2，set#2')
y = tf.matmul(x,W) +b
y_ = tf.placeholder("float",[None,1])

#3
print('\n#3，strain_step')
cost = tf.reduce_sum(tf.pow((y_-y),2))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

#4
print('\n#4，init')
init = tf.initialize_all_variables()

#5
print('\n#5，Session')
ss = tf.Session()
ss.run(init)

#6
for i in range(1000):
    #6a
    x_s = np.random.rand(1,1)
    y_s = np.dot([[0.33]],np.random.rand(1,1)) + 0.33
    #6b
    feed = {x: x_s, y_: y_s}
    ss.run(train_step,feed_dict=feed)
    #6c
    wdat=ss.run(W)
    bdat=ss.run(b)
    wd2,x2,y2=wdat[0][0],x_s[0][0],y_s[0][0]
    bd2=bdat[0]
    #print(x2,y2)
    print('{0}#,W:{1:.4}, b:{2:.4} ,x_s:{3:.4f} ,y_s:{4:.4f}'.format(i,wd2,bd2,x2,y2))
    #
    #print("After %d iteration:"%i)
    #print("W : %f"%ss.run(W))
    #print("b : %f"%ss.run(b))
    #6d
    #print(x_s,y_s)
    #xsum = np.concatenate((xsum,x_s))
    #ysum = np.concatenate((ysum,y_s))
    #pd


#7
'''
print('\n#7.1,xsum')
print(xsum)
print('\n#7.2,ysum')
print(ysum)
'''
#8
print('\n#7.1,xsum')
print(df.tail())