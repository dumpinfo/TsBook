import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Cnn_Engine(object):
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
        print('build convolutional neural network')
        X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        self.model['X'] = X
        y = tf.placeholder(shape=[None, self.k], dtype=tf.float32)
        self.model['y'] = y
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        self.model['keep_prob'] = keep_prob
        # 第2层第1个卷积层c1
        W_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0.0, stddev=0.1))
        self.model['W_1'] = W_1
        b_2 = tf.Variable(tf.zeros([28, 28, 6]))
        self.model['b_2'] = b_2
        z_2 = tf.nn.conv2d(X, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_2
        self.model['z_2'] = z_2
        a_2 = tf.nn.relu(z_2)
        self.model['a_2'] = a_2
        # 第3层第1个最大池化层
        m_3 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.model['m_3'] = m_3
        # 第4层第2个卷积层C2
        W_4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0.0, stddev=0.1))
        self.model['W_4'] = W_4
        b_4 = tf.Variable(tf.zeros([10, 10, 16]))
        self.model['b_4'] = b_4
        z_4 = tf.nn.conv2d(m_3, W_4, strides=[1, 1, 1, 1], padding='VALID') + b_4
        self.model['z_4'] = z_4
        a_4 = tf.nn.relu(z_4)
        self.model['a_4'] = a_4
        # 第5层第2个最大池化层
        m_5 = tf.nn.max_pool(a_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.model['m_5'] = m_5
        # 第6层第3个卷积层
        W_5 = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 120], mean=0.0, stddev=0.1))
        self.model['W_5'] = W_5
        b_6 = tf.Variable(tf.zeros([1, 1, 120]))
        self.model['b_6'] = b_6
        z_6_raw = tf.nn.conv2d(m_5, W_5, strides=[1, 1, 1, 1], padding='VALID') + b_6
        self.model['z_6_raw'] = z_6_raw
        z_6 = tf.reshape(z_6_raw, [-1, 120])
        self.model['z_6'] = z_6
        a_6 = tf.nn.relu(z_6)
        self.model['a_6'] = a_6
        a_6_dropout = tf.nn.dropout(a_6, keep_prob)
        self.model['a_6_dropout'] = a_6_dropout
        # 第7层第1个全连接层
        W_6 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0.0, stddev=0.1))
        self.model['W_6'] = W_6
        b_7 = b_6 = tf.Variable(tf.zeros([84]))
        self.model['b_7'] = b_7
        z_7 = tf.matmul(a_6_dropout, W_6) + b_7
        self.model['z_7'] = z_7
        a_7 = tf.nn.relu(z_7)
        self.model['a_7'] = a_7
        a_7_dropout = tf.nn.dropout(a_7, keep_prob)
        self.model['a_7_dropout'] = a_7_dropout
        # 第8层第2个全连接层
        W_7 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0.0, stddev=0.1))
        self.model['W_7'] = W_7
        b_8 = tf.Variable(tf.zeros([10]))
        self.model['b_8'] = b_8
        z_8 = tf.matmul(a_7_dropout, W_7) + b_8
        self.model['z_8'] = z_8
        y_ = tf.nn.softmax(z_8)
        self.model['y_'] = y_
        #训练部分
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), 
                                        reduction_indices=[1]))
        self.model['cross_entropy'] = cross_entropy
        #train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        loss = cross_entropy + self.lanmeda*(tf.reduce_sum(W_1**2) + 
                tf.reduce_sum(W_4**2) + 
                tf.reduce_sum(W_5**2) + 
                tf.reduce_sum(W_6**2) + 
                tf.reduce_sum(W_7**2))
        self.model['loss'] = loss
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(loss)
        self.model['train_step'] = train_step
        correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))
        self.model['correct_prediction'] = correct_prediction
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.model['accuracy'] = accuracy
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
            if Cnn_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                for batch_idx in range(total_batch):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    X_mb_raw, y_mb = mnist.train.next_batch(self.batch_size)
                    X_mb = X_mb_raw.reshape([self.batch_size, 28, 28, 1])
                    sess.run(train_step, feed_dict={X: X_mb, y: y_mb, 
                            keep_prob: self.keep_prob})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        eval_runs += 1
                        eval_times.append(eval_runs)
                        train_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_train.reshape([-1, 28, 28, 1]), 
                                y: y_train, keep_prob: 1.0})
                        train_accs.append(train_accuracy)
                        validation_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_validation.reshape([-1, 28, 28, 1]), 
                                            y: y_validation, 
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
            print(sess.run(accuracy, feed_dict={X: X_test.reshape([-1, 28, 28, 1]),
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
        sample = np.array(raw)
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        #sample = X_test[102]
        X_run = sample.reshape(1, 28, 28, 1)
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
            # W_1
            W_1 = sess.run(self.model['W_1'], feed_dict={X: X_run, keep_prob: 1.0})
            print('W_1:{0}'.format(W_1))
            a_2 = sess.run(self.model['a_2'], feed_dict={X: X_run, keep_prob: 1.0})
            fm1 = a_2[0, :, :, 0]
            fm2 = a_2[0, :, :, 1]
            fm3 = a_2[0, :, :, 2]
            fm4 = a_2[0, :, :, 3]
            fm5 = a_2[0, :, :, 4]
            fm6 = a_2[0, :, :, 5]
        img_in = sample.reshape(28, 28)
        plt.figure(1)
        plt.subplot(241)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
        # feature map1
        plt.subplot(242)
        plt.imshow(fm1, cmap='gray')
        plt.axis('off')
        plt.title('fm1')
        # feature map2
        plt.subplot(243)
        plt.imshow(fm2, cmap='gray')
        plt.axis('off')
        plt.title('fm2')
        # feature map3
        plt.subplot(244)
        plt.imshow(fm3, cmap='gray')
        plt.axis('off')
        plt.title('fm3')
        # feature map4
        plt.subplot(246)
        plt.imshow(fm4, cmap='gray')
        plt.axis('off')
        plt.title('fm4')
        # feature map5
        plt.subplot(247)
        plt.imshow(fm5, cmap='gray')
        plt.axis('off')
        plt.title('fm5')
        # feature map6
        plt.subplot(248)
        plt.imshow(fm6, cmap='gray')
        plt.axis('off')
        plt.title('fm6')
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

