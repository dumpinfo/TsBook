#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

pkmital案例合集

@from:
pkmital案例合集网址：
https://github.com/pkmital/tensorflow_tutorials

'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#----------------------------

#1
print('\n#1,创建一些玩具的数据')
rlog='/ailib/log_tmp'
#plt.ioff()
#plt.ion() #使用交互绘图模式
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)

#2
print('\n#2,绘制scatter散点图')
ax.scatter(xs, ys)
fig.show()
plt.draw()

#3
print('\n#3,设置x，y作为tf.placeholders占位符变量，用于graph图计算数据输入')
X = tf.placeholder(tf.float32,name='X')
Y = tf.placeholder(tf.float32,name='Y')

#4
print('\n#4,我们将创建一个多次多项式函数，代替单因子W权重和和bias偏离参数')
print('不断调整模型输入： (X^0, X^1, X^2, ...)，和最终的输出（Y）')
Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
for pow_i in range(1, 5):
    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)


#5
print('\n#5,Loss损失函数，也称cost代价函数')
print('用于衡量预测数值和实际数据之间的差距')
cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_observations - 1)

#6
print('\n#7,设置learning_rate学习速率，用于optimizer优化函数')
print('我们将通过迭代，不断降低用Loss损失函数（cost代价函数）的数值，提高模型预测精度')
print('我们也可以添加正则化，不过需要增加其他方面的代价')
print('例如：使用岭回归的收缩控制参数，增加模型精度')
print('     理论上，收缩值越大，模型越健壮，使用以下函数')
print('cost = tf.add(cost, tf. multiply (1e-6, tf.global_norm([W])))')
print(' ')
print('使用梯度下降算法，优化W、B参数')
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#7
print('\n#7,创建graph图计算的session变量')
print('设置迭代次数，n_epochs = 1000')
n_epochs = 1000
with tf.Session() as sess:
    #7.1
    print('\n#7.1,初始化所有graph图计算的所有变量')
    print('使用summary日志函数，保存graph图计算结构图')
    sess.run(tf.global_variables_initializer())
    xsum= tf.summary.FileWriter(rlog, sess.graph) 
    
    #7.2
    print('\n#7.2,使用全部训练数据，进行模型学习')
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        #7.3
        #print('\n#7.3,使用xs，ys训练模型')
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})
            
        #6.4
        #print('\n#6.4,计算每次迭代后的损失函数数值')
        training_cost = sess.run(
            cost, feed_dict={X: xs, Y: ys})
        
        #7.5
        if epoch_i % 50 == 0:
            print('\n#7.5,绘制中间效果图')
            print(epoch_i,'#',training_cost)
            #fig.show()
            ax.plot(xs, Y_pred.eval(
                feed_dict={X: xs}, session=sess),
                    'k', alpha=epoch_i / n_epochs)
            fig.show()
            plt.draw()
            
        #7.6
        dk=np.abs(prev_training_cost - training_cost)
        if dk < 0.000001:
            print('\n#7.6,模型误差值间隔小于0.000001，可以提前退出训练过程')
            print('\n@{0}#,dk,{1:.8f}'.format(epoch_i,dk))
            break
        
        #7.7
        #print('\n#7.7,保存训练误差到变量prev_training_cost')
        prev_training_cost = training_cost

    #8        
    print('\n#8,显示最终结果')
    ys2=Y_pred.eval(feed_dict={X: xs}, session=sess)
    ax.plot(xs, ys2,'k')
    fig.show()
