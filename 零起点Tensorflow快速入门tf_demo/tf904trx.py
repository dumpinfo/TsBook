#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn
#import sklearn
from sklearn.cross_validation import train_test_split  
#from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt

#-----------------
""" Linear Regression Example """



# Regression data
#X = [3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]
#Y = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]
fss='data/p100.csv'
df=pd.read_csv(fss)
print(df.tail())
#X,Y=df['x'].values,df['y'].values
#X,Y=df['x'].tolist,df['y'].tolist
X,Y=list(df['x'].values),list(df['y'].values)
#print(X)
print(type(X))
# Linear Regression graph
input_ = tflearn.input_data(shape=[None])
linear = tflearn.single_unit(input_)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.001)
m = tflearn.DNN(regression, tensorboard_verbose=3)
#m = tflearn.DNN(regression,logdir='/tmp/tflearn_logs', tensorboard_verbose=3)
#m = tflearn.Trainer(train_ops=m_op,tensorboard_dir='/tmp/tflearn_logs/',tensorboard_verbose=2)
    # Training for 10 epochs.
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) +
      "*X + " + str(m.get_weights(linear.b)))

print("\nTest prediction for x = 3.2, 3.3, 3.4:")
ys2=m.predict(X)
#print(m.predict([3.2, 3.3, 3.4]))
# should output (close, not exact) y = [1.5315033197402954, 1.5585315227508545, 1.5855598449707031]

#8.1
print('\n#7,plot')
plt.plot(X,Y, 'ro', label='Original data')
#8.2

plt.plot(X, ys2, label='ln_model')
#
#8.3
plt.legend()
plt.show()


