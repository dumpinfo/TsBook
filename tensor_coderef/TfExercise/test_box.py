# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""

from __future__ import print_function
# 导入numpy库， numpy是一个常用的科学计算库，优化矩阵的运算
import numpy as np
np.random.seed(1337)
import csv

# 导入mnist数据库， mnist是常用的手写数字库
# 导入顺序模型
from keras.models import Sequential
# 导入全连接层Dense， 激活层Activation 以及 Dropout层
from keras.layers.core import Dense, Dropout, Activation




# 设置batch的大小
batch_size = 100
# 设置类别的个数
#nb_classes = 10
# 设置迭代的次数
nb_epoch = 1000

'''
下面这一段是加载mnist数据，网上用keras加载mnist数据都是用
(X_train, y_train), (X_test, y_test) = mnist.load_data()
但是我用这条语句老是出错：OSError: [Errno 22] Invalid argument
'''
#from tensorflow.examples.tutorials.mnist import input_data  
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
sample_size = int(50000)
X_train = np.zeros((sample_size,4096))
Y_train = np.zeros((sample_size,4))

csvFile = open("train_ft.txt", "r")
reader = csv.reader(csvFile)  # 返回的是迭代类型
data_x = []
for item in reader:
    #print(item)
    data_x.append(item)
csvFile.close()

print("---------------reading x finishing--------------------\n")


for ii in range(0, sample_size):
    for jj in range(0, 4096):
       X_train[ii][jj] = float(data_x[ii][jj])

print("---------------filling x finishing--------------------\n")

csvFile = open("list.csv", "r")
reader = csv.reader(csvFile)  # 返回的是迭代类型
data_y = []
for item in reader:
    #print(item)
    data_y.append(item)
csvFile.close()

print("---------------reading y finishing--------------------\n")

for ii in range(0, sample_size):
    for jj in range(0, 4):
       Y_train[ii][jj] = float(data_y[ii][jj+1])

print("---------------filling y finishing--------------------\n")

X_test = X_train[0:2000,:]
Y_test = Y_train[0:2000,:]
#X_train, Y_train = mnist.train.images,mnist.train.labels  
#X_test, Y_test = mnist.test.images, mnist.test.labels  
#X_train = X_train.reshape(-1, 28, 28,1).astype('float32')  
#X_test = X_test.reshape(-1,28, 28,1).astype('float32')  
print("#######################################################\n")
print("#################START TRAINING########################\n")
print("#######################################################\n")
print("#######################################################\n")
#打印训练数据和测试数据的维度
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
print("#######################################################\n")
#修改维度
X_train = X_train.reshape(sample_size,4096)
X_test = X_test.reshape(2000,4096)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)



# keras中的mnist数据集已经被划分成了55,000个训练集，10,000个测试集的形式，按以下格式调用即可

# X_train原本是一个60000*28*28的三维向量，将其转换为60000*4096的二维向量

# X_test原本是一个10000*28*28的三维向量，将其转换为10000*4096的二维向量

# 将X_train, X_test的数据格式转为float32存储
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
#X_train /= 255
#X_test /= 255
# 打印出训练集和测试集的信息
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)

# 建立顺序型模型
model = Sequential()
'''
模型需要知道输入数据的shape，
因此，Sequential的第一层需要接受一个关于输入数据shape的参数，
后面的各个层则可以自动推导出中间数据的shape，
因此不需要为每个层都指定这个参数
''' 

# 输入层有4096个神经元
# 第一个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(2048, input_shape=(4096,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 第二个隐层有1600个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(1600))
model.add(Activation('relu'))
model.add(Dropout(0.15))

# 第三个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.05))

# 输出层有10个神经元，激活函数为SoftMax，得到分类结果
model.add(Dense(4))
model.add(Activation('relu'))

# 输出模型的整体信息
# 总共参数数量为4096*512+512 + 512*512+512 + 512*10+10 = 669706
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size = 200,
                    epochs = 80000,
                    verbose = 1,
                    validation_data = (X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

model.save_weights("model.h5")
print("Saved model to disk")

# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])
