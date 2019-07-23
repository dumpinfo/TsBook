#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用keras简化接口 IRNN循环神经网络模型
IRNN：Initialize Recurrent Networks of Rectified Linear Units

本案例900次迭代学习，可以达到93%的识别率
原来需要1687500（168w）次迭代学习。 
每次迭代运行时间，290秒，e3-1230  CPU

@from:
http://keras.io


'''


import keras,os
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
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

batch_size = 32
num_classes = 10
epochs = 10 #200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

#2.1
print('\n#2.1,get.mnist.dat & xed')
x_train, y_train, x_val, y_val, x_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

#2.2
x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#2.3
print('\n#2.3,x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#3
print('\n#2,将类向量转换成二进制矩阵')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#4 
print('\n#4,构建IRNN神经网络模型')
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)

#5
print('\n#5,输出网络模型参数model.summary')
#tl.network.print_layers()
model.summary()

#6
print('\n#6,编译网络模型model.compile')
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

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
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
