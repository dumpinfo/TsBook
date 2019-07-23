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
#import sklearn
from sklearn.cross_validation import train_test_split  

import matplotlib.pyplot as plt

#-----------------

#1
print('\n#1,dat_init')
rlog='/ailib/log_tmp'
learingRate = 0.001
num_epoch=1000
num_split=0.4
#
fss='data/p100.csv'
df=pd.read_csv(fss)
print(df.tail())

#2
print('\n#2,data.xed')
xs,ys=df['x'].values,df['y'].values
dnum=len(df['x'])
#print('dnum，',dnum)
#print('type(xs)',type(xs))


#3
print('\n#3,set.var')
with tf.name_scope('dat-set'):
    x = tf.placeholder(tf.float32,name='x')
    y = tf.placeholder(tf.float32,name='y')
    W = tf.Variable(0.8,name='weight')
    b = tf.Variable(0.5,name='bias')

#4 y=x*W+b
print('\n#4,model')
with tf.name_scope('mode'):
    model = tf.add(tf.multiply(x, W),b,name='model_y')
    #model=x * W +b

#5
print('\n#5,fun')
with tf.name_scope('fun.xxx'):
    loss = tf.reduce_sum(tf.pow(model-y, 2))/(2*dnum)  #L2 loss
    #loss=tf.reduce_sum(tf.pow(model-y,2))
    #
    #梯度下降法，learningRate = 0.001，目标是最小化loss变量
    #optimizer = tf.train.GradientDescentOptimizer(learingRate)
    #train = optimizer.minimize(loss)
    train = tf.train.GradientDescentOptimizer(learingRate).minimize(loss)
    
#6
print('\n#6,summary')
with tf.name_scope('summary'):
    #6.1
    for value in [x, y,W, b,loss]:
        tf.summary.scalar(value.op.name, value)
        
    #6.2
    tf.summary.histogram('weight', W)
    tf.summary.histogram('loss', loss)
    
    #6.3
    summary_all = tf.summary.merge_all()


#7
print('\n#7,Session')

with tf.name_scope('Session'):
    #7.1
    ss=tf.Session()
    xsum= tf.summary.FileWriter(rlog, ss.graph)  
    
    #7.2
    init = tf.global_variables_initializer()
    ss.run(init)
    
    #7.3
    saver = tf.train.Saver()
    saver.restore(ss, 'tmp/m001.ckpt')  
    
    #7.4
    w2=ss.run(W)
    b2=ss.run(b)
    dss='w,{0:.4f},  b{1:.4f}'.format(w2,b2)
    print(dss)
    #---------
#

#
#8.1
print('\n#7,plot')
plt.plot(xs, ys, 'ro', label='Original data')
#8.2
ys2=ss.run(model,feed_dict={x:xs}) 
plt.plot(xs, ys2, label='ln_model')
#
#8.3
plt.legend()
plt.show()

#9
print('\n#9,df')
ys3=ss.run(W) * xs + ss.run(b)
df['ys2']=ys2
df['ys3']=ys3
print(df.tail())
ss.close()

