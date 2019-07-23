#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

线性回归 
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
print('\n#2,set 输入数据')
X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]

#3
print('\n#3,构建线性回归神经网络模型')
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
    metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression,tensorboard_dir=rlog)

#4
print('\n#4,开始训练模型')
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

#5
print('\n#5,输出模型训练的结果参数：')
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

#6
print('\n#6,根据模型，进行预测')
print("\nTest prediction for x = 3.2, 3.3, 3.4:")
print(m.predict([3.2, 3.3, 3.4]))

