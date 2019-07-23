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

#-----------------
rlog='/ailib/log_tmp'
# 使用 NumPy 生成假数据(phony data), 总共 1000 个点.
xs = np.float32(np.random.rand(2, 1000)) # 随机输入
ys = np.dot([0.200, 0.300], xs) + 0.400
# 构造一个线性模型
x = tf.placeholder(tf.float32,name='x')
W = tf.Variable(tf.zeros([1]),name='W')
b = tf.Variable(tf.zeros([1]),name='b')

#b = tf.Variable(tf.zeros([1]))
#W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
#y = tf.matmul(W, xs) + b
y = W * xs + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - ys))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# 初始化变量

#4
print('\n#4,summar.misc')
with tf.name_scope('summar'):
    for value in [x, W, b, y, loss]:
        tf.summary.scalar(value.op.name, value)
        #print(value.op.name, value)
     
        

#
m10 = tf.summary.merge_all()
#
# 启动图 (graph)
ss = tf.Session()
init = tf.initialize_all_variables()
ss.run(init)

#6
print('\n#6,summary，rlog',rlog)
xsum= tf.summary.FileWriter(rlog, ss.graph)  

# 
for i in range(0, 100):
    mdat=ss.run(m10)
    xsum.add_summary(mdat, i)
    #
    
    ss.run(train)
    if i % 10 == 0:
        #print (step, sess.run(W), sess.run(b))
        w_dat,b_dat=ss.run(W),ss.run(b)
        #xdat=ss.run(lost, feed_dict=feed)
        xdat=ss.run(loss)
        #xdat=ss.run(loss,feed_dict={x:xs, y:ys})
        w2,b2=w_dat[0],b_dat[0]
        
        dss='{0} #, w,{1:.4f}, b,{2:.4f} ,x,{3:.4f} '.format(i,w2,b2,xdat)
        #print(step, ss.run(W), ss.run(b))
        print(dss)
     
        
'''
#1
print('\n#1,set.dat')
with tf.name_scope('data-set'):
    rlog='/ailib/log_tmp'
    
    x = tf.placeholder(tf.float32,name='x')
    W = tf.Variable(tf.zeros([1]),name='W')
    b = tf.Variable(tf.zeros([1]),name='b')
    y_ = tf.placeholder(tf.float32,name='y')

#2
print('\n#2,model')
with tf.name_scope('model'):
    y = W * x + b
    

print('\n#2,model')
with tf.name_scope('fun'):
    lost = tf.reduce_mean(tf.square(y_-y))
    optimizer = tf.train.GradientDescentOptimizer(0.0000001)
    train_step = optimizer.minimize(lost)

#4
print('\n#4,summar.misc')
with tf.name_scope('summar'):
    for value in [x, W, b, y_, lost]:
        tf.summary.scalar(value.op.name, value)
        #print(value.op.name, value)
     

#
m10 = tf.summary.merge_all()
#merged = tf.summary.merge_all()  
#train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)  
#test_writer = tf.summary.FileWriter(log_dir + '/test')  

#5
print('\n#5,Session')
with tf.Session() as ss:
    xsum= tf.summary.FileWriter(rlog, ss.graph)  
    #
    tf.global_variables_initializer().run()  
               
    
    #6
    print('\n#6,summary，rlog',rlog)
    #xsum= tf.summary.FileWriter(rlog, ss.graph)  

    #7
    print('\n#7,Session init')
    #init = tf.global_variables_initializer()
    #ss.run(init)

    #8
    print('\n#8,train')
    steps = 1000
    for i in range(steps):
        #8.1
        #mdat, acc = sess.run([m10, accuracy], feed_dict=feed_dict(False))
        #mdat, acc = ss.run([m10, accuracy], feed_dict=feed)
        #mdat=ss.run(m10)
        #xsum.add_summary(mdat, i)
        
        #8.2    
        xs = [i]
        ys = [3 * i]
        feed = { x: xs, y_: ys }
        ss.run(train_step, feed_dict=feed)
        
        #
        #mdat, acc = ss.run([m10, accuracy], feed_dict=feed)
        #mdat, acc = ss.run([m10, accuracy], feed_dict=feed)
        mdat=ss.run(m10)
        
        #8.3
        if i % 10 == 0 :
            w_dat,b_dat=ss.run(W),ss.run(b)
            xdat=ss.run(lost, feed_dict=feed)
            w2,b2=w_dat[0],b_dat[0]
            #
            dss='{0} #, w,{1:.4f}, b,{2:.4f} ,x,{3:.4f} '.format(i,w2,b2,xdat)
            print(dss,acc)

    #9
    #print('\n#9,close')
    #ss.close
    xsum.close()
'''