#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用keras简化接口 

本案例20次迭代学习，可以达到98.40%的识别率
每次迭代运行时间，7秒，e3-1230  CPU
每次迭代运行时间，2秒，K520 GPU.

@from:
http://keras.io
'''


import keras,os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
#
import tensorlayer as tl
import tensorflow as tf


#-------------------
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)
#
batch_size = 128
num_classes = 10
epochs = 5 # 20

#2
print('\n#2,get.mnist.dat')
x_train, y_train, x_val, y_val, x_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

x_train /= 255 #0.0,...,1.0,gray
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#3
print('\n#2,将类向量转换成二进制矩阵')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#4 
print('\n#4,构建神经网络算法模型')
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#5
print('\n#5,输出网络模型参数model.summary')
#tl.network.print_layers()
model.summary()


#6
print('\n#6,编译网络模型model.compile')
model.compile(loss='categorical_crossentropy',
	optimizer=RMSprop(),metrics=['accuracy'])

#7
print('\n#7,TensorBoard回调补丁callbacks')
tbCallBack = keras.callbacks.TensorBoard(log_dir=rlog, histogram_freq=0, write_graph=True, write_images=True)
#...
#model.fit(...inputs and parameters..., callbacks=[tbCallBack])


#8
print('\n#8,开始训练模型')
history = model.fit(x_train, y_train,batch_size=batch_size,
                    epochs=epochs,verbose=1,validation_data=(x_test, y_test),
                    callbacks=[tbCallBack])
#history = model.fit(x_train, y_train,batch_size=batch_size,
#                    epochs=epochs,verbose=1,validation_data=(x_test, y_test))

#
#9 
print('\n#9,evaluate评估模型训练效果')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
