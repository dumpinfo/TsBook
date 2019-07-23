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
print('\n#2,set 输入数据，逻辑异或运算：XOR')
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_nand = [[1.], [1.], [1.], [0.]]
Y_or = [[0.], [1.], [1.], [1.]]


#3.1
print('\n#3.1,对应模型相关的参数、函数')
g = tflearn.input_data(shape=[None, 2])
# Nand operator definition
g_nand = tflearn.fully_connected(g, 32, activation='linear')
g_nand = tflearn.fully_connected(g_nand, 32, activation='linear')
g_nand = tflearn.fully_connected(g_nand, 1, activation='sigmoid')
g_nand = tflearn.regression(g_nand, optimizer='sgd',
                            learning_rate=2.,
                            loss='binary_crossentropy')
# Or operator definition
g_or = tflearn.fully_connected(g, 32, activation='linear')
g_or = tflearn.fully_connected(g_or, 32, activation='linear')
g_or = tflearn.fully_connected(g_or, 1, activation='sigmoid')
g_or = tflearn.regression(g_or, optimizer='sgd',
                          learning_rate=2.,
                          loss='binary_crossentropy')
# XOR merging Nand and Or operators
g_xor = tflearn.merge([g_nand, g_or], mode='elemwise_mul')

#3.2
print('\n#3.2,定义模型')
m = tflearn.DNN(g_xor,tensorboard_dir=rlog)

#4
print('\n#4,开始训练模型')
m.fit(X, [Y_nand, Y_or], n_epoch=400, snapshot_epoch=False)


#5
print('\n#5,测试模型')
print("Testing XOR operator")
print("0 xor 0:", m.predict([[0., 0.]]))
print("0 xor 1:", m.predict([[0., 1.]]))
print("1 xor 0:", m.predict([[1., 0.]]))
print("1 xor 1:", m.predict([[1., 1.]]))

