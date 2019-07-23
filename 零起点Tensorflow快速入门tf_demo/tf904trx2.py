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
import tflearn
#import sklearn
#from sklearn.cross_validation import train_test_split  

import matplotlib.pyplot as plt

#-----------------

#1 
print('\n#1,set data')
num_epoch=500
fss='data/p100.csv'
df=pd.read_csv(fss)
print(df.tail())

#2 
with tf.name_scope('data_xs_ys'):
    xs,ys=list(df['x'].values),list(df['y'].values)
    print(type(xs))

with tf.name_scope('var'):
    #x = tflearn.input_data(shape=[None],name='x')
    x = tflearn.input_data(shape=[None],name='x')
    #x = tf.placeholder(tf.float32,name='x')
    #y = tf.placeholder(tf.float32,name='y')
    #W = tf.Variable(0.8,name='weight')
    #b = tf.Variable(0.5,name='bias')
    
with tf.name_scope('fun'):
    linear = tflearn.single_unit(x,name='linear')
    regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',metric='R2', learning_rate=0.001)

with tf.name_scope('model'):
    #W3 = tf.Variable(0.5,tf.random_normal([256, 10]), name="W3")
    #b3 = tf.Variable(tf.random_normal([10]), name="b3")
    #x = tf.add(tf.matmul(x, W3), b3)
    #mx = tf.add(tf.multiply(x, W),b,name='model_y')
    mx = tflearn.DNN(regression, tensorboard_verbose=0)
    mx.fit(xs,ys,n_epoch=num_epoch,show_metric=True, snapshot_epoch=False)

with tf.name_scope('get'):
    w2=mx.get_weights(linear.W)
    b2=mx.get_weights(linear.b)
    #print(w2)
    dss='y={0:.4}*x+{1:.4}'.format(w2[0],b2[0])
    print(dss)

print('\n#1,predict')
with tf.name_scope('predict'):
#print("\nTest prediction for x = 3.2, 3.3, 3.4:")
    ys2=mx.predict(xs)
    print(ys2)
    
    
#
#8.1
print('\n#7,plot')
plt.plot(xs,ys, 'ro', label='Original data')
#8.2

plt.plot(xs, ys2, label='ln_model')
#
#8.3
plt.legend()
plt.show()



'''
#1
print('\n#1,dat_init')
rlog='/ailib/log_tmp'
learingRate = 0.001
num_epoch=500
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
    for xc in range(num_epoch):
        #7.4
        x2,x2t,y2,y2t= train_test_split(xs, ys, test_size=0.4, random_state=0)
        ss.run(train,feed_dict={x:x2,y:y2}) 
        
        #7.5
        if(xc % 50 == 0):
            #7.6
            #xdat=ss.run(summary_all,feed_dict={x:xs,y:ys}) #!!!,error
            xdat=ss.run(summary_all,feed_dict={x:x2[0],y:y2[0]})
            xsum.add_summary(xdat, xc)
            
            #7.7
            l2=ss.run(loss,feed_dict={x:xs,y:ys})
            w2=ss.run(W)
            b2=ss.run(b)
            dss='{0}#,loss,{1:.4f},  w,{2:.4f},  b{2:.4f}'.format(xc,l2,w2,b2)
            print(dss)
                #---------

    
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

'''