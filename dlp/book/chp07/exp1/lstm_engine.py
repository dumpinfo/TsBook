import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from app_global import FLAGS

class Lstm_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self):
        self.datasets_dir = 'datasets/'
        self.input_vec_size = self.lstm_size = 28 # 输入向量的维度
        self.time_step_size = 28 # 循环层长度
        self.batch_size = 128
        self.test_size = 256
        self.num_category = 10
        
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/lgr.ckpt'):
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        self.build_model()
        epochs = 100
        saver = tf.train.Saver()
        check_interval = 50 # 50
        best_accuracy = -0.01
        improve_threthold = 1.005
        no_improve_steps = 0
        max_no_improve_steps = 200 #3000
        is_early_stop = False
        eval_runs = 0
        eval_times = []
        train_accs = []
        validation_accs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Lstm_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                batch_idx = 1
                for start, end in zip(range(0, len(X_train), self.batch_size),
                                range(self.batch_size, len(X_train)+1,
                                self.batch_size)):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    sess.run(self.train_op, feed_dict={self.X:
                                    X_train[start:end],
                                    self.y: y_train[start:end]})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        eval_runs += 1
                        eval_times.append(eval_runs)
                        #train_accuracy = sess.run(self.accuracy,
                        #                feed_dict={self.X: X_train,
                        #                self.y: y_train})
                        train_accuracy = self.calculate_accuracy(sess,
                                       X_train, y_train)
                        train_accs.append(train_accuracy)
                        #validation_accuracy = sess.run(self.accuracy,
                        #                feed_dict={self.X: X_validation,
                        #                self.y: y_validation})
                        validation_accuracy = self.calculate_accuracy(sess,
                                       X_validation, y_validation)
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
                    batch_idx += 1
            print('test result:{0}'.format(sess.run(self.accuracy,
                            feed_dict={self.X: X_test, self.y: y_test})))
            plt.figure(1)
            plt.subplot(111)
            plt.plot(eval_times, train_accs, 'b-', label='train accuracy')
            plt.plot(eval_times, validation_accs, 'r-', 
                    label='validation accuracy')
            plt.title('accuracy trend')
            plt.legend(loc='lower right')
            plt.show()
        
    def run(self, ckpt_file='work/lgr.ckpt'):
        img_file = 'datasets/test5.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        sample = np.array(raw)
        self.build_model()
        X_run = sample.reshape(1, 28, 28)
        saver = tf.train.Saver()
        digit = -1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_file)
            rst = sess.run(self.y_, feed_dict={self.X: X_run})
            digit = np.argmax(rst)
        img_in = sample.reshape(28, 28)
        plt.figure(1)
        plt.subplot(111)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
        plt.show()
        
    def calculate_accuracy(self, sess, X, y):
        # 计算随机批次上的精度
        indices = np.arange(len(X))  # Get A Test Batch
        np.random.shuffle(indices)
        indices = indices[0:self.batch_size]
        return np.mean(np.argmax(y[indices], axis=1) ==
                         np.argmax(sess.run(self.y_, 
                         feed_dict={self.X: X[indices]}), axis=1))
        
    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.time_step_size, self.input_vec_size])
        XT = tf.transpose(self.X, [1, 0, 2])
        XR = tf.reshape(XT, [-1, self.lstm_size])
        X_split = tf.split(XR, self.time_step_size, 0)
        self.y = tf.placeholder("float", [None, self.num_category])
        W = tf.Variable(tf.random_normal(shape=[self.lstm_size, self.num_category], mean=0.0, stddev=0.01))
        b = tf.Variable(tf.random_normal(shape=[self.num_category], mean=0.0, stddev=0.0001))
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=True)
        z, _states = tf.contrib.rnn.static_rnn(lstm, X_split, dtype=tf.float32)
        y_ = tf.matmul(z[-1], W) + b
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=self.y))
        self.train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
        self.correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.y_ = y_
        
    def load_datasets(self):
        ''' 调用Tensorflow的input_data，读入MNIST手写数字识别数据集的
        训练样本集、验证样本集、测试样本集
        '''
        mnist = input_data.read_data_sets(self.datasets_dir, 
                one_hot=True)
        raw_X_train = mnist.train.images
        X_train = raw_X_train.reshape(-1, self.input_vec_size, self.input_vec_size)
        y_train = mnist.train.labels
        raw_X_validation = mnist.validation.images
        X_validation = raw_X_validation.reshape(-1, self.input_vec_size, self.input_vec_size)
        y_validation = mnist.validation.labels
        raw_X_test = mnist.test.images
        X_test = raw_X_test.reshape(-1, self.input_vec_size, self.input_vec_size)
        y_test = mnist.test.labels
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist