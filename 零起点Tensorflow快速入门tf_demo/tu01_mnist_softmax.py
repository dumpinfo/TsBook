#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例

@from:
A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners

'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#------------------

#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

#2
print('\n#2,构建模型')
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

#3
print('\n#3,定义loss损失函数和optimizer优化函数')
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#4
print('\n#4,Session')
#sess = tf.InteractiveSession()
sess = tf.Session()

#tf.global_variables_initializer().run()
init = tf.global_variables_initializer()
sess.run(init)
#
xsum= tf.summary.FileWriter(rlog, sess.graph)  

#5
print('\n#5,Train，nstep=1500')
nstep=1500
for xc in range(nstep):
    #5.a set step#n,dat,run
    
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed = {x: batch_xs, y_: batch_ys}
    xdat=sess.run(train_step, feed_dict=feed)
  
    #5.b print.info
    if xc % 100==0:
        b_dat=sess.run(b)
        xdat=sess.run(cross_entropy, feed_dict=feed)
        b2=b_dat[0]
          
        #
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        adat=sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})
        #
        dss='{0} #, b,{1:.4f} ,x,{2:.4f},k,{3:.4f} '.format(xc,b2,xdat,adat)
        print(dss)
          
        #print(xc,'#,acc,',xdat)
      

#6
print('\n#6,Test测试模型训练结果')
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
adat=sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})
print('acc,',adat)

#6
print('\n#6,Session.close')
sess.close()


