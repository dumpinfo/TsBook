#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用TensorLayer简化接口 

@from:
http://tensorlayer.org

'''


import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import time,os

#------------------------------------------

#1
print('\n#1,Session')
#rlog='/ailib/log_tmp'
rlog='logs/'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)
sess = tf.InteractiveSession()


#2
print('\n#2,set.dat')
X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
X_val = np.asarray(X_val, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)

print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_val.shape', X_val.shape)
print('y_val.shape', y_val.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))

#3 
print('\n#3,定义placeholder占位符参数')


# placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

#4 
print('\n#4,构建神经网络算法模型')
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
network = tl.layers.DenseLayer(network, n_units=10,
			act = tf.identity,name='output')

#5 
print('\n#5,# 打印神经网络各层的属性参数')
attrs = vars(network)
print(', '.join("%s: %s\n" % item for item in attrs.items()))
print('\nnetwork.all_drop')
print(network.all_drop)     # {'drop1': 0.8, 'drop2': 0.5, 'drop3': 0.5}
print('\nnetwork.all_layers')    
print(network.all_layers) 

#6 
print('\n#6,# 定义cost损失函数和衡量指标，SOFTMAX多项式回归模型，使用的是tl模块内置函数，以提高速度')
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)


#7 
print('\n#7,优化cost代价函数')

# You can add more penalty to the cost function as follow.
# cost = cost + tl.cost.maxnorm_regularizer(1.0)(network.all_params[0]) + tl.cost.maxnorm_regularizer(1.0)(network.all_params[2])
# cost = cost + tl.cost.lo_regularizer(0.0001)(network.all_params[0]) + tl.cost.lo_regularizer(0.0001)(network.all_params[2])
# cost = cost + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[0]) + tl.cost.maxnorm_o_regularizer(0.001)(network.all_params[2])

#8 
print('\n#8,初始化全部变量参数，定义Optimizer优化函数')
params = network.all_params
# train
n_epoch = 5  #100
batch_size = 128
learning_rate = 0.0001
print_freq = 5
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost)

tl.layers.initialize_global_variables(sess)

#9 
print('\n#9.1,输出网络模型变量参数')
network.print_params()
#
print('\n#9.2,输出网络模型参数')
network.print_layers()
#
print('\n#9.3,其他参数')
print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)

#10 
print('\n#10,迭代模式，训练模型')
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=batch_size, n_epoch=n_epoch, print_freq=1,
            X_val=X_val, y_val=y_val, eval_train=False,
            tensorboard =True,tensorboard_weight_histograms=True,tensorboard_graph_vis=True)


#11 
print('\n#11,评估模型训练效果')
test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(
                            X_test, y_test, batch_size, shuffle=True):
    dp_dict = tl.utils.dict_to_one( network.all_drop )    # disable noise layers
    feed_dict = {x: X_test_a, y_: y_test_a}
    feed_dict.update(dp_dict)
    err, ac = sess.run([cost, acc], feed_dict=feed_dict)
    test_loss += err; test_acc += ac; n_batch += 1
print("   test loss: %f" % (test_loss/n_batch))
print("   test acc: %f" % (test_acc/n_batch))

#12 
print('\n#12,保存训练好的模型数据')
saver = tf.train.Saver()
save_path = saver.save(sess, "tmp/model.ckpt")
print("Model saved in file: %s" % save_path)


# You can also save the parameters into .npz file.
tl.files.save_npz(network.all_params , name='tmp/model.npz')
# You can only save one parameter as follow.
# tl.files.save_npz([network.all_params[0]] , name='model.npz')
# Then, restore the parameters as follow.
# load_params = tl.files.load_npz(path='', name='model.npz')
# tl.files.assign_params(sess, load_params, network)

#13
print('\n#13,Session.close')
sess.close()
