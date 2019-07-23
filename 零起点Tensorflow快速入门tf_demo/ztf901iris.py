#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import pdb
import tensorflow as tf
import tflearn

import numpy as np
import pandas as pd
from tflearn.data_utils import load_csv

#-----------------

pdb.Restart()

#1
print('\n#1,set.dat')



# Load IRIS dataset
data, labels = load_csv("data/iris4x.csv", categorical_labels=True, n_classes=3)
fss='data/iris2.csv'
df=pd.read_csv(fss)
print(df.tail())

#===============
# Build neural network, one hidden layer, 7 units (faster than 3 layers of 10,20,10 units)
net = tflearn.input_data(shape=[None, 4])
net = tflearn.fully_connected(net, 7)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, learning_rate=0.1)

# Define model
model = tflearn.DNN(net)

# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=40, show_metric=True, validation_set=0.2)

# do some predictions
pred = model.predict([[5.1, 3.5, 1.4, 0.2], [6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]])
#print([5.1, 3.5, 1.4, 0.2], "is most probably a", find_class(pred[0]))
#print([6.4, 3.2, 4.5, 1.5], "is most probably a", find_class(pred[1]))
#print([5.8, 3.1, 5.0, 1.7], "is most probably a", find_class(pred[2]))
print('\npred,',pred)

# Results
# Training Step: 80  | total loss: 0.30226
#| Adam | epoch: 040 | loss: 0.30226 - acc: 0.9566 | val_loss: 0.22871 - val_acc: 0.9667 -- iter: 120/120
#--
#[0.9860564470291138, 0.013943483121693134, 1.5376224382634973e-07]
#[5.1, 3.5, 1.4, 0.2] is a setosa
#[0.04990610480308533, 0.7952917218208313, 0.15480217337608337]
#[6.4, 3.2, 4.5, 1.5] is a versicolor
#[0.0037880234885960817, 0.2782661020755768, 0.7179458737373352]
#[5.8, 3.1, 5.0, 1.7] is a virginica

