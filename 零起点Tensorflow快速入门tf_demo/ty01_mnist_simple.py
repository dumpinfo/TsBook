#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用TensorLayer简化接口  SOFTMAX多项式回归模型

@from:
http://tensorlayer.org

'''




import tensorflow as tf
import tensorlayer as tl

#------------------

#
#1
print('\n#1,Session')
#rlog='/ailib/log_tmp'
rlog='logs/'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)
sess = tf.InteractiveSession()
#sess = tf.Session()

#2
print('\n#2,set.dat')
X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1,784), path="data/mnist/")

#3 
print('\n#3,定义placeholder占位符参数')
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')


#4 
print('\n#4,构建神经网络算法模型')
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=800,
                                act = tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')


#5 
print('\n#5,# 定义损失函数和衡量指标，SOFTMAX多项式回归模型，使用的是tl模块内置函数，以提高速度')
print('参见：tf.nn.sparse_softmax_cross_entropy_with_logits() 实现 softmax')
network = tl.layers.DenseLayer(network, n_units=10,
                                act = tf.identity,
                                name='output_layer')

#6 
print('\n#6,定义cost代价函数')
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

#7 
print('\n#7,定义optimizer优化函数')

train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

#8 
print('\n#8,初始化全部变量参数')
tl.layers.initialize_global_variables(sess)


#9 
print('\n#9.1,输出网络模型变量参数')
network.print_params()  ##sess = tf.InteractiveSession()
#
print('\n#9.2,输出网络模型参数')
network.print_layers()



#10 
print('\n#10,迭代模式，训练模型')
knum=10 #500

'''
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=500, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)
'''
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=knum, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False,
            tensorboard =True,tensorboard_weight_histograms=True,tensorboard_graph_vis=True)

#11 
print('\n#11,评估模型训练效果')
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)

#12 
print('\n#12,保存训练好的模型数据，使用.npz文件格式')
tl.files.save_npz(network.all_params , name='tmp/model.npz')

#13
print('\n#13,Session.close')
sess.close()
