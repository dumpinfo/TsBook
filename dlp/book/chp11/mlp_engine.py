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
    
    def __init__(self, rbms, datasets_dir):
        self.datasets_dir = datasets_dir
        self.batch_size = 100
        self.n = 784
        self.k = 10
        self.L = np.array([self.n, 1024, 784, 512, 256, self.k])
        self.lanmeda = 0.001
        self.keep_prob_val = 0.75
        self.rbms = rbms
        self.model = {}
                
    def build_model(self, mode='train'):
        print('mode={0}'.format(mode))
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        if 'train' == mode:
            # 取出预训练去噪自动编码机参数
            with self.rbms[0].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae0_W1 = sess.run(self.rbms[0].W)
                    dae0_b2 = sess.run(self.rbms[0].bh_)
            with self.rbms[1].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae1_W1 = sess.run(self.rbms[1].W)
                    dae1_b2 = sess.run(self.rbms[1].bh_)
            with self.rbms[2].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae2_W1 = sess.run(self.rbms[2].W)
                    dae2_b2 = sess.run(self.rbms[2].bh_)
            with self.rbms[3].tf_graph.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    dae3_W1 = sess.run(self.rbms[3].W)
                    dae3_b2 = sess.run(self.rbms[3].bh_)
        # 从784到1024的去噪自动编码机
        if 'train' == mode:
            self.W_1 = tf.Variable(dae0_W1, name='W_1')
            self.b_2 = tf.Variable(dae0_b2, name='b_2')
        else:
            self.W_1 = tf.Variable(tf.truncated_normal([784, 1024], mean=0.0,
                stddev=0.1), name='W_1')
            self.b_2 = tf.Variable(tf.zeros([1024]), name='b_2')
        self.z_2 = tf.matmul(self.X, self.W_1) + self.b_2
        self.a_2 = tf.nn.tanh(self.z_2) # tf.nn.relu(self.z_2)
        self.a_2_dropout = tf.nn.dropout(self.a_2, self.keep_prob)
        # 从1024到784的去噪自动编码机
        if 'train' == mode:
            self.W_2 = tf.Variable(dae1_W1, name='W_2')
            self.b_3 = tf.Variable(dae1_b2, name='b_3')
        else:
            self.W_2 = tf.Variable(tf.truncated_normal([1024, 784], mean=0.0,
                stddev=0.1), name='W_2')
            self.b_3 = tf.Variable(tf.zeros([784]), name='b_3')
        self.z_3 = tf.matmul(self.a_2_dropout, self.W_2) + self.b_3
        self.a_3 = tf.nn.tanh(self.z_3) # tf.nn.relu(self.z_3)
        self.a_3_dropout = tf.nn.dropout(self.a_3, self.keep_prob)
        # 从784到512的去噪自动编码机
        if 'train' == mode:
            self.W_3 = tf.Variable(dae2_W1, name='W_3')
            self.b_4 = tf.Variable(dae2_b2, name='b_4')
        else:
            self.W_3 = tf.Variable(tf.truncated_normal([784, 512], mean=0.0,
                stddev=0.1), name='W_3')
            self.b_4 = tf.Variable(tf.zeros([512]), name='b_4')
        self.z_4 = tf.matmul(self.a_3_dropout, self.W_3) + self.b_4
        self.a_4 = tf.nn.tanh(self.z_4) #tf.nn.relu(self.z_4)
        self.a_4_dropout = tf.nn.dropout(self.a_4, self.keep_prob)
        # 从512到256的去噪自动编码机
        if 'train' == mode:
            self.W_4 = tf.Variable(dae3_W1, name='W_4')
            self.b_5 = tf.Variable(dae3_b2, name='b_5')
        else:
            self.W_4 = tf.Variable(tf.truncated_normal([512, 256], mean=0.0,
                stddev=0.1), name='W_4')
            self.b_5 = tf.Variable(tf.zeros([256]), name='b_5')
        self.z_5 = tf.matmul(self.a_4_dropout, self.W_4) + self.b_5
        self.a_5 = tf.nn.tanh(self.z_5) #tf.nn.relu(self.z_5)
        self.a_5_dropout = tf.nn.dropout(self.a_5, self.keep_prob)
        #输出层
        self.W_5 = tf.Variable(tf.zeros([256, 10]))
        self.b_6 = tf.Variable(tf.zeros([10]))
        self.z_6 = tf.matmul(self.a_5_dropout, self.W_5) + self.b_6
        self.y_ = tf.nn.softmax(self.z_6)
        #训练部分
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(
                        self.y * tf.log(self.y_), 
                        reduction_indices=[1]))
        #train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        self.loss = self.cross_entropy + self.lanmeda*(
                tf.reduce_sum(self.W_1**2) + 
                tf.reduce_sum(self.W_2**2) + tf.reduce_sum(self.W_3**2) + 
                tf.reduce_sum(self.W_4**2) + tf.reduce_sum(self.W_5**2))
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9,
                beta2=0.999, epsilon=1e-08, use_locking=False,
                name='Adam').minimize(self.loss)
        self.correct_prediction = tf.equal(tf.arg_max(self.y_, 1),
                        tf.arg_max(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                        tf.float32))
        return self.X, self.y_, self.y, self.keep_prob, self.cross_entropy, \
                self.train_step, self.correct_prediction, self.accuracy
    
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/mlp.ckpt'):
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
                            keep_prob: self.keep_prob_val})
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
        
    def run(self, ckpt_file='work/mlp.ckpt'):
        print('run..........')
        img_file = 'datasets/test5.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        #sample = np.array(raw)
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        sample = X_test[102]
        X_run = sample.reshape(1, 784)
        digit = -1
        with tf.Graph().as_default():
            X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                    accuracy = self.build_model(mode='run')
            saver = tf.train.Saver()
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
                W_1 = sess.run(self.W_1)
                wight_map = W_1[:,0].reshape(28, 28)
                a_2 = sess.run(self.a_2, feed_dict={X: X_run, \
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

