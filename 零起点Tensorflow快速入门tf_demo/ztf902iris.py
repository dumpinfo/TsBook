#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''


#import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------



#1
print('\n#1,set.dat')



# Load IRIS dataset
#data, labels = load_csv("data/iris4x.csv", categorical_labels=True, n_classes=3)
fss='data/iris2.csv'
df=pd.read_csv(fss)


print(df.tail())
#ipd = pd.read_csv("iris.csv")

#===============
species = list(df['xid'].unique())
df['One-hot'] = df['xid'].map(lambda x: np.eye(len(species))[species.index(x)] )
df.sample(5)
print(df.tail())
#

shuffled = df.sample(frac=1)
trainingSet = shuffled[0:len(shuffled)-50]
testSet = shuffled[len(shuffled)-50:]


inp = tf.placeholder(tf.float32, [None, 4])
weights = tf.Variable(tf.zeros([4, 3]))
bias = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(inp, weights) + bias)

y_ = tf.placeholder(tf.float32, [None, 3])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#
#keys = ['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width']
keys = ['x1', 'x2','x3', 'x4']
for i in range(1000):
    train = trainingSet.sample(50)
    sess.run(train_step, feed_dict={inp: [x for x in train[keys].values],
                                    y_: [x for x in train['One-hot'].as_matrix()]})

xss=sess.run(accuracy, feed_dict={inp: [x for x in testSet[keys].values], y_: [x for x in testSet['One-hot'].values]})
print('xss,',xss) 


def classify(inpv):
    dim = y.get_shape().as_list()[1]
    res = np.zeros(dim)
    # argmax returns a single element vector, so get the scalar from it
    largest = sess.run(tf.argmax(y,1), feed_dict={inp: inpv})[0]
    return np.eye(dim)[largest]
    
sample = shuffled.sample(1)
print('sample',sample)
print( "Classified as %s" , classify(sample[keys]))


#=============
'''
TensorFlow and Iris http://tneal.org/post/tensorflow-iris/TensorFlowIris/

cors=['','r','g','b']
for xc in range(1,4):
    css,cor,ksiz='xid_'+str(xc),cors[xc],xc*10
    df2=df[df['xid']==xc]
    if xc==1:
        ax = df2.plot.scatter(x='x1', y='x2', color=cor,label=css, s=ksiz);
    else:
        df2.plot.scatter(x='x1', y='x2',  color=cor, label=css, s=ksiz, ax=ax);
'''