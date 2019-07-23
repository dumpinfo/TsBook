#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

#An example showing how to save/restore models and retrieve weights. 
保存和读取神经网络模型
使用TFlearn简化接口 

@from:
tflearn.org
'''

import os
import tflearn
import tensorflow as tf
import tflearn.datasets.mnist as mnist


#------------------
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)

#2
print('\n#2,set MNIST输入数据')
X, Y, testX, testY = mnist.load_data(data_dir='data/mnist/',one_hot=True)

#3.1
print('\n#3.1,构建线性回归神经网络模型')
input_layer = tflearn.input_data(shape=[None, 784], name='input')
dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
softmax = tflearn.fully_connected(dense2, 10, activation='softmax')
regression = tflearn.regression(softmax, optimizer='adam',
    learning_rate=0.001,loss='categorical_crossentropy')

#3.2
print('\n#3.2,设置数据目录：checkpoint (autosave)检查点，summary日志')
model = tflearn.DNN(regression, checkpoint_path='tmp/model.tfl.ckpt',tensorboard_dir=rlog)

#4
print('\n#4,开始训练模型')
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
          snapshot_step=50, #500 Snapshot (save & evalaute) model every 500 steps.
          run_id='model_and_weights')


#5
print('\n#5,保存训练好的模型')
model.save("tmp/model.tfl")

#6
print('\n#6,读取训练好的模型')
print('也可以读取中间checkpoint检查点的模型数据')
print('例如：model.load("model.tfl.ckpt-500")')
model.load("tmp/model.tfl")


#7
print('\n#7,利用读取的模型，再次进行训练')
model.fit(X, Y, n_epoch=1,
          validation_set=(testX, testY),
          show_metric=True,
          snapshot_epoch=True,
          run_id='model_and_weights')


#8
print('\n#8,检查模型权重数据')
print('\n#8.1,使用神经网络图层名称，读取权重数据')

dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
print('\n#8.1,使用神经网络图层名称，读取权重数据')
print("使用get_weights函数，获取稠密层#1权重数据Dense1 layer weights:")
print(model.get_weights(dense1_vars[0]))
print("使用get_value通用函数，获取稠密层#1偏移值数据Dense1 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense1_vars[1]))

print('\n#8.2,使用简化属性：w、b，读取权重数据')
print("使用get_weights函数，获取稠密层#2权重数据Dense1 layer weights:")
print(model.get_weights(dense2.W))
print("使用get_value通用函数，获取稠密层#1偏移值数据Dense1 layer biases:")
with model.session.as_default():
    print(tflearn.variables.get_value(dense2.b))
