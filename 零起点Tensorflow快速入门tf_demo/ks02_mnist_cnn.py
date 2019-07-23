#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用keras简化接口 cnn卷积神经网络

本案例12次迭代学习，可以达到99.25%的识别率
每次迭代运行时间，138秒，e3-1230 CPU
每次迭代运行时间，16秒，K520 GPU.

@from:
http://keras.io
'''


import keras,os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorlayer as tl
import tensorflow as tf
#
#-------------------
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

#2.1
print('\n#2.1,get.mnist.dat & xed')
x_train, y_train, x_val, y_val, x_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

#2.2
#K.image_data_format(),channels_last
print('\n#2.2, K.image_data_format()',K.image_data_format())
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#2.3
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#2.4
print('\n#2.4,x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#3
print('\n#2,将类向量转换成二进制矩阵')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#4 
print('\n#4,构建神经网络算法模型')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#5
print('\n#5,输出网络模型参数model.summary')
#tl.network.print_layers()
model.summary()

#6
print('\n#6,编译网络模型model.compile')
model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

#7
print('\n#7,TensorBoard回调补丁callbacks')
tbCallBack = keras.callbacks.TensorBoard(log_dir=rlog, histogram_freq=0, write_graph=True, write_images=True)
#...
#model.fit(...inputs and parameters..., callbacks=[tbCallBack])

#8
print('\n#8,开始训练模型')
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
		verbose=1,validation_data=(x_test, y_test),callbacks=[tbCallBack])
		
#9 
print('\n#9,evaluate评估模型训练效果')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
