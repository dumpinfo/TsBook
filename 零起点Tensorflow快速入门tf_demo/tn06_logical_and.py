#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

超智能体：逻辑运算
使用TFlearn简化接口 


@from:
tflearn.org
'''
import os
import tflearn
import tensorflow as tf

#------------------
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)


#2
print('\n#2,set 输入数据，逻辑与运算：AND')
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [0.], [0.], [1.]]

#3.1
print('\n#3.1,对应模型相关的参数、函数')
g = tflearn.input_data(shape=[None, 2])
g = tflearn.fully_connected(g, 128, activation='linear')
g = tflearn.fully_connected(g, 128, activation='linear')
g = tflearn.fully_connected(g, 1, activation='sigmoid')
g = tflearn.regression(g, optimizer='sgd', learning_rate=2.,loss='mean_square')

#3.2
print('\n#3.2,定义模型')
m = tflearn.DNN(g,tensorboard_dir=rlog)

#4
print('\n#4,开始训练模型')
m.fit(X, Y, n_epoch=100, snapshot_epoch=False)

#5
print('\n#5,测试模型')
print("Testing AND operator")
print("0 and 0:", m.predict([[0., 0.]]))
print("0 and 1:", m.predict([[0., 1.]]))
print("1 and 0:", m.predict([[1., 0.]]))
print("1 and 1:", m.predict([[1., 1.]]))
