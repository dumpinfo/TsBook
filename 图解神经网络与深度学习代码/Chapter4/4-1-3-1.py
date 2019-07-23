# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#"xyz坐标内的点"
xy=tf.placeholder(tf.float32,[None,2])
z=tf.placeholder(tf.float32,[None,1])
#"初始化z=w1*x+w2*y+b中的w1、w2、b"
w=tf.Variable(tf.constant([[1],[1]],tf.float32),dtype=tf.float32)
b=tf.Variable(1.0,dtype=tf.float32)
#"损失函数"
loss=tf.reduce_sum(tf.square(z-(tf.matmul(xy,w)+b)))
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
#"梯度下降法"
opti=tf.train.GradientDescentOptimizer(0.005).minimize(loss)
#"记录每一次迭代后的平均平方误差(Mean Squared Error)"
MSE=[]
#"训练数据"
xy_train=np.array([
                [1,1],
                [2,1],
                [3,2],
                [1,2],
                [4,5],
                [5,8]
                ],np.float32)
z_train=np.array([
                [8],
                [12],
                [10],
                [14],
                [28],
                [10]
                ],np.float32)
#"声明一个tf.train.Saver类"
saver=tf.train.Saver()
#"训练模型,循环500次"
for i in range(500):
    #"梯度下降"
    session.run(opti,feed_dict={xy:xy_train,z:z_train})
    #"计算每一次迭代的损失值，并追加到列表中进行保存"
    MSE.append(session.run(loss,feed_dict={xy:xy_train,z:z_train}))
    #"每隔100次打印 w和b的值"
    if i%100==0:
        #"保存模型"
        saver.save(session,
                './regression/regressionModel.ckpt',global_step=i)
        print('------"第"'+str(i)+'"次的迭代值"-------')
        print(session.run([w,b]))
#"打印第500次（最后)的迭代值"
print('------"第" '+str(500)+'"次的迭代值"-------')
print(session.run([w,b]))
saver.save(session,'./regression/regressionModel.ckpt',global_step=i)
#"画出损失函数的值"
plt.figure(1)
plt.plot(MSE)
plt.show()