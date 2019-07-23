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

x = tf.placeholder(tf.float32,name='x')
W = tf.Variable(tf.zeros([1]),name='W')
b = tf.Variable(tf.zeros([1]),name='b')
y_ = tf.placeholder(tf.float32,name='y')

#2
print('\n#2,model')
y = W * x + b

#3
print('\n#3,fun...l')
lost = tf.reduce_mean(tf.square(y_-y))
optimizer = tf.train.GradientDescentOptimizer(0.0000001)
train_step = optimizer.minimize(lost)

#4
print('\n#4,summar.misc')
for value in [x, W, b, y_, lost]:
    tf.summary.scalar(value.op.name, value)
    #print(value.op.name, value)
 
m10 = tf.summary.merge_all()
#print(m10[0])
#5
print('\n#5,Session')
ss = tf.Session()

#6
print('\n#6,summary，rlog',rlog)
xsum= tf.summary.FileWriter(rlog, ss.graph)  

#7
print('\n#7,Session init')
init = tf.global_variables_initializer()
ss.run(init)

#8
print('\n#8,train')
steps = 1000
for i in range(steps):
    #8.1
    #mdat=ss.run(m10)
    #xsum.add_summary(mdat, i)
    
    #8.2    
    xs = [i]
    ys = [3 * i]
    feed = { x: xs, y_: ys }
    ss.run(train_step, feed_dict=feed)
    
    #8.3
    if i % 10 == 0 :
        w_dat,b_dat=ss.run(W),ss.run(b)
        xdat=ss.run(lost, feed_dict=feed)
        w2,b2=w_dat[0],b_dat[0]
        #
        dss='{0} #, w,{1:.4f}, b,{2:.4f} ,x,{3:.4f} '.format(i,w2,b2,xdat)
        print(dss)

#9
print('\n#9,close')
ss.close
xsum.close()
