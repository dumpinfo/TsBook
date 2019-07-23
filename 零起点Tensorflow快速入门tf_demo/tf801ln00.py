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
import matplotlib.pyplot as plt

#-----------------

#1
rlog='/ailib/log_tmp'

#xs = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#ys = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
#print(type(xs))
df=pd.read_csv('data/dat_ln.csv')
xs=df['x'].values
ys=df['y'].values
dnum=len(df['x'])
#print(dnum)

with tf.name_scope('dat-set'):
    x = tf.placeholder(tf.float32,name='x')
    y = tf.placeholder(tf.float32,name='y')
    
    #rng = np.random
    #W = tf.Variable(rng.randn(),name="weight")
    #b = tf.Variable(rng.randn(),name="bais")
    W = tf.Variable(0.8,name='weight')
    b = tf.Variable(0.,name='bias')

#参数
learingRate = 0.01
with tf.name_scope('mode'):
    model = tf.add(tf.multiply(x, W),b,name='model')

with tf.name_scope('fun.xxx'):
    #model = tf.add(tf.multiply(x, W),b)
    loss = tf.reduce_sum(tf.pow(model-y, 2))/(2*dnum)#l2 loss
    #梯度下降法，learningRate = 0.01，目标是最小化loss变量
    #train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    #
    #loss = tf.reduce_mean(tf.square(model - y))
    #loss = tf.reduce_sum(tf.pow(model-y, 2)) #/(2*dnum)#l2 loss
    optimizer = tf.train.GradientDescentOptimizer(learingRate)
    train = optimizer.minimize(loss)
#
with tf.name_scope('summar'):
    for value in [x, y,W, b,loss]:
        tf.summary.scalar(value.op.name, value)
    #
    tf.summary.histogram('histogram', W)
    tf.summary.histogram('loss', loss)
    #
    m10 = tf.summary.merge_all()

#


with tf.name_scope('Session-wrk'):
#with tf.Session() as ss:
    ss=tf.Session()
    #xsum= tf.summary.FileWriter(rlog, ss.graph)  
    #
    init = tf.global_variables_initializer()
    ss.run(init)
    xsum= tf.summary.FileWriter(rlog, ss.graph)  
    #
    print (ss.run(model,feed_dict={x:xs}))
    for xc in range(500):
        #8.a
        
        #
        #xdat=ss.run(summaries,feed_dict={x:X})
        #
        for (X,Y) in zip(xs,ys):
            ss.run(train,feed_dict={x:X,y:Y}) 
            #_,xdat=ss.run([train,summaries],  feed_dict={x:X,y:Y})
            #xdat=ss.run(summaries,  feed_dict={x:X,y:Y})
            xdat=ss.run(m10,  feed_dict={x:X,y:Y})
            xsum.add_summary(xdat, xc)
            #
            
        if(xc % 50 == 0):
            #xdat=ss.run(summaries)
            #xsum.add_summary(xdat, xc)
            #
            #
            l2=ss.run(loss,feed_dict={x:xs,y:ys})
            
            w2=ss.run(W)
            b2=ss.run(b)
            dss='{0}#,loss,{1:.4f},  w,{2:.4f},  b{2:.4f}'.format(xc,l2,w2,b2)
            print(dss)
            #
            #print( "Step: ","d"%step," cost= ","{:.9f}".format(sess.run(loss,feed_dict={x:train_X,y:train_Y}))\
            #," W= ",sess.run(W)," b= ",sess.run(b))
            #---------
    #        
    print (ss.run(model,feed_dict={x:xs}))
    #
    plt.plot(xs, ys, 'ro', label='Original data')
    ys2=ss.run(W) * xs + ss.run(b)
    plt.plot(xs, ys2, label='Fitted line')
    plt.legend()
    plt.show()
