#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用keras简化接口 分层HRNN循环神经网络模型
 HRNN:Hierarchical RNN 循环神经网络模型

本案例5次迭代学习，可以达到98%的识别率
每次迭代运行时间，1200秒，e3-1230 CPU
每次迭代运行时间，16秒，K520 GPU.

@from:
http://keras.io
'''


import keras,os
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM


import tensorlayer as tl
import tensorflow as tf
#
#-------------------
#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)

# Training parameters.
batch_size = 32
num_classes = 10
epochs = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

#2.1
print('\n#2.1,get.mnist.dat & xed')
x_train, y_train, x_val, y_val, x_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

#2.2
# Reshapes data to 4D for Hierarchical RNN.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#2.3
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#2.4
print('\n#2.4,x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#3.1
print('\n#3.1,将类向量转换成二进制矩阵')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row, col, pixel = x_train.shape[1:]

#3.2
print('\n#3.2,设置4D输入向量参数')
x = Input(shape=(row, col, pixel))

#3.3
print('\n#3.3,使用TimeDistributeds时间分布函数，对行像素进行编码')
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)


#3.4
print('\n#3.4,使用LSTM函数，对列像素进行编码')
encoded_columns = LSTM(col_hidden)(encoded_rows)

#4.1 
print('\n#4,构建模型prediction预测函数')
prediction = Dense(num_classes, activation='softmax')(encoded_columns)

#4.2 
print('\n#4.2,构建神经网络算法模型')
model = Model(x, prediction)

#5
print('\n#5,输出网络模型参数model.summary')
#tl.network.print_layers()
model.summary()

#6
print('\n#6,编译网络模型model.compile')
model.compile(loss='categorical_crossentropy',
	optimizer='rmsprop',metrics=['accuracy'])

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
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
