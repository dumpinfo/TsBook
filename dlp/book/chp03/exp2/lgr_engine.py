import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Lgr_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir
        self.batch_size = 100
    
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/lgr.ckpt'):
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        X, W, b, y_, y, cross_entropy, train_step, correct_prediction, \
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
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Lgr_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                for batch_idx in range(total_batch):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    X_mb, y_mb = mnist.train.next_batch(self.batch_size)
                    sess.run(train_step, feed_dict={X: X_mb, y: y_mb})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        train_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_train, y: y_train})
                        validation_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_validation, y: y_validation})
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
                                      y: y_test}))        
        
    def run(self, ckpt_file='work/lgr.ckpt'):
        img_file = 'datasets/test6.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        sample = np.array(raw)
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        X, W, b, y_, y, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        #sample = X_test[102]
        X_run = sample.reshape(1, 784)
        saver = tf.train.Saver()
        digit = -1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_file)
            rst = sess.run(y_, feed_dict={X: X_run})
            print('rst:{0}'.format(rst))
            max_prob = -0.1
            for idx in range(10):
                if max_prob < rst[0][idx]:
                    max_prob = rst[0][idx]
                    digit = idx;
        img_in = sample.reshape(28, 28)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
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
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist
                
    def build_model(self):
        X = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        #y_ = tf.matmul(X, W) + b
        z = tf.matmul(X, W) + b
        y_ = tf.nn.softmax(z)
        y = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        lanmeda = 0.001
        J = cross_entropy + lanmeda*tf.reduce_sum(W**2)
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
        #       cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(J)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X, W, b, y_, y, cross_entropy, train_step, \
                correct_prediction, accuracy
































