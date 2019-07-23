import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Mlp_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir
        self.batch_size = 100
        self.n = 784
        self.k = 10
        self.L = np.array([self.n, 512, self.k])
        self.lanmeda = 0.001
        self.keep_prob = 0.75
        self.model = {}
                
    def build_model(self):
        relu_node = 1
        if 1 == relu_node:
            return self.build_relu()
        else:
            return self.build_sigmoid()
        
    def build_relu(self):
        print('###### relu #####')
        X = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        #隐藏层
        W_1 = tf.Variable(tf.truncated_normal([784, 512], mean=0.0, 
                stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
        b_2 = tf.Variable(tf.zeros([512])) #隐含层偏置b1全部初始化为0
        z_2 = tf.matmul(X, W_1) + b_2
        a_2 = tf.nn.relu(z_2)
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        a_2_dropout = tf.nn.dropout(a_2, keep_prob)
        #输出层
        W_2 = tf.Variable(tf.zeros([512, 10]))
        b_3 = tf.Variable(tf.zeros([10]))
        z_3 = tf.matmul(a_2_dropout, W_2) + b_3
        y_ = tf.nn.softmax(z_3)
        #训练部分
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), 
        reduction_indices=[1]))
        #train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        loss = cross_entropy + self.lanmeda*(tf.reduce_sum(W_1**2) + 
                tf.reduce_sum(W_2**2))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(loss)
        correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.saveModelTensor(X, y, W_1, b_2, z_2, a_2, W_2, b_3, z_3, y_, 
                cross_entropy, loss, train_step, correct_prediction, accuracy)
        return X, y_, y, keep_prob, cross_entropy, train_step, \
                correct_prediction, accuracy
                
    def saveModelTensor(self, X, y, W_1, b_2, z_2, a_2, W_2, b_3, z_3, y_, 
            cross_entropy, loss, train_step, correct_prediction, accuracy):
        # 保存模型
        self.model['X'] = X
        self.model['y'] = y
        self.model['W_1'] = W_1
        self.model['b_2'] = b_2
        self.model['z_2'] = z_2
        self.model['a_2'] = a_2
        self.model['W_2'] = W_2
        self.model['b_3'] = b_3
        self.model['z_3'] = z_3
        self.model['y_'] = y_
        self.model['cross_entropy'] = cross_entropy
        self.model['loss'] = loss
        self.model['train_step'] = train_step
        self.model['correct_prediction'] = correct_prediction
        self.model['accuracy'] = accuracy

    
    def build_sigmoid(self):
        print('###### sigmoid #####')
        self.keep_prob = 0.90
        X = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        # 隐藏层
        W_1 = tf.Variable(tf.random_normal(shape=[784, 512], mean=0.0, 
                stddev=1.0)) # W_t
        b_2 = tf.Variable(tf.zeros(shape=[512]))
        z_2 = tf.matmul(X, W_1) + b_2
        a_2 = tf.nn.sigmoid(z_2)
        # 输出层
        W_2 = tf.Variable(tf.random_normal(shape=[512, 10], 
                mean=0.0, stddev=1.0)) # W_t
        b_3 = tf.Variable(tf.random_normal(shape=[10], mean=0.0, 
                stddev=1.0))
        z_3 = tf.matmul(a_2, W_2) + b_3
        y_ = tf.nn.softmax(z_3)
        # 代价函数
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
        cross_entropy = tf.reduce_sum(- y * tf.log(y_), 1)
        loss = tf.reduce_mean(cross_entropy)
        #train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(loss)
        # 精度计算
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X, y_, y, keep_prob, cross_entropy, train_step, \
                correct_prediction, accuracy
    
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/lgr.ckpt'):
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        epochs = 10
        saver = tf.train.Saver()
        total_batch = int(mnist.train.num_examples/self.batch_size)
        check_interval = 50
        best_accuracy = -0.01
        improve_threthold = 1.005
        no_improve_steps = 0
        max_no_improve_steps = 3000
        is_early_stop = False
        eval_runs = 0
        eval_times = []
        train_accs = []
        validation_accs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Mlp_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                for batch_idx in range(total_batch):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    X_mb, y_mb = mnist.train.next_batch(self.batch_size)
                    sess.run(train_step, feed_dict={X: X_mb, y: y_mb, 
                            keep_prob: self.keep_prob})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        eval_runs += 1
                        eval_times.append(eval_runs)
                        train_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_train, y: y_train, keep_prob: 1.0})
                        train_accs.append(train_accuracy)
                        validation_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_validation, y: y_validation, 
                                keep_prob: 1.0})
                        validation_accs.append(validation_accuracy)
                        if best_accuracy < validation_accuracy:
                            if validation_accuracy / best_accuracy >= \
                                    improve_threthold:
                                no_improve_steps = 0
                            best_accuracy = validation_accuracy
                            saver.save(sess, ckpt_file)
                        print('{0}:{1}# train:{2}, validation:{3}'.format(
                                epoch, batch_idx, train_accuracy, 
                                validation_accuracy))
            print(sess.run(accuracy, feed_dict={X: X_test,
                                      y: y_test, keep_prob: 1.0}))
            plt.figure(1)
            plt.subplot(111)
            plt.plot(eval_times, train_accs, 'b-', label='train accuracy')
            plt.plot(eval_times, validation_accs, 'r-', 
                    label='validation accuracy')
            plt.title('accuracy trend')
            plt.legend(loc='lower right')
            plt.show()
        
    def run(self, ckpt_file='work/lgr.ckpt'):
        img_file = 'datasets/test6.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        #sample = np.array(raw)
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        sample = X_test[102]
        X_run = sample.reshape(1, 784)
        saver = tf.train.Saver()
        digit = -1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_file)
            rst = sess.run(y_, feed_dict={X: X_run, keep_prob: 1.0})
            print('rst:{0}'.format(rst))
            max_prob = -0.1
            for idx in range(10):
                if max_prob < rst[0][idx]:
                    max_prob = rst[0][idx]
                    digit = idx
            # W_1_1
            W_1 = sess.run(self.model['W_1'])
            wight_map = W_1[:,0].reshape(28, 28)
            a_2 = sess.run(self.model['a_2'], feed_dict={X: X_run, \
                    keep_prob: 1.0})
            a_2_raw = a_2[0]
            a_2_img = a_2_raw[0:484]
            feature_map = a_2_img.reshape(22, 22)
        img_in = sample.reshape(28, 28)
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(wight_map, cmap='gray')
        plt.axis('off')
        plt.title('wight row')
        plt.subplot(133)
        plt.imshow(feature_map, cmap='gray')
        plt.axis('off')
        plt.title('hidden layer')
        plt.show()
        
    def load_datasets(self):
        ''' 调用Tensorflow的input_data，读入MNIST手写数字识别数据集的
        训练样本集、验证样本集、测试样本集
        '''
        mnist = input_data.read_data_sets(self.datasets_dir, 
                one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        '''
        print('X_train: {0} y_train:{1}'.format(
                X_train.shape, y_train.shape))
        print('X_validation: {0} y_validation:{1}'.format(
                X_validation.shape, y_validation.shape))
        print('X_test: {0} y_test:{1}'.format(
                X_test.shape, y_test.shape))
        image_raw = (X_train[1] * 255).astype(int)
        image = image_raw.reshape(28, 28)
        label = y_train[1]
        idx = 0
        for item in label:
            if 1 == item:
                break
            idx += 1
        plt.title('digit:{0}'.format(idx))
        plt.imshow(image, cmap='gray')
        plt.show()
        '''
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist

