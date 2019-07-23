import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from app_global import FLAGS
from tensorflow.examples.tutorials.mnist import input_data

class Dae_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self):
        self.datasets_dir = 'datasets/'
        self.random_seed = 1 # 用于测试目的，使每次生成的随机数相同
        self.input_data = None
        self.input_labels = None
        self.keep_prob = None
        self.layer_nodes = []  # list of layers of the final network
        self.train_step = None
        self.cost = None
        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.loss_func = 'cross_entropy'
        self.enc_act_func = tf.nn.tanh
        self.dec_act_func = tf.nn.tanh
        self.num_epochs = 10
        self.batch_size = 10
        self.opt = 'adam' # 优化算法：sgd, adagrand, momentum
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.corr_type = 'masking' # 噪音方式：none, salt_and_pepper, masking
        self.corr_frac = 0.1 # 加入噪音比例
        self.regtype = 'l2' # 调整技术类别：none l1, l2
        self.regcoef = 5e-4 # 调整项系数
        self.n = 784 # 输入向量维度
        self.hidden_size = 1024 # 隐藏层大小
        pass
        
    def run(self, ckpt_file='work/dae.ckpt'):
        img_file = 'datasets/test5.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        sample = np.array(raw)
        X_run = sample.reshape(1, 784)
        digit = -1
        with self.tf_graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt_file)
                hidden_data, output_data = sess.run([self.a2, self.y_], 
                                feed_dict={self.X: X_run})
        img_in = sample.reshape(28, 28)
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img_in, cmap='gray')
        plt.title('origin')
        plt.axis('off')
        # 
        plt.subplot(132)
        hidden_pic = hidden_data.reshape(32, 32)
        plt.imshow(hidden_pic, cmap='gray')
        plt.axis('off')
        plt.title('hidden layer')
        # 
        plt.subplot(133)
        restore_img = output_data.reshape(28, 28)
        plt.imshow(restore_img, cmap='gray')
        plt.axis('off')
        plt.title('restore image')
        plt.show()
        
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/dae.ckpt'):
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        with self.tf_graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.num_epochs):
                    X_train_prime = self.add_noise(sess, X_train, self.corr_frac)
                    shuff = list(zip(X_train, X_train_prime))
                    np.random.shuffle(shuff)
                    batches = [_ for _ in self.gen_mini_batches(shuff, self.batch_size)]
                    batch_idx = 1
                    for batch in batches:
                        X_batch_raw, X_prime_batch_raw = zip(*batch)
                        X_batch = np.array(X_batch_raw).astype(np.float32)
                        X_prime_batch = np.array(X_prime_batch_raw).astype(np.float32)
                        batch_idx += 1
                        opv, loss = sess.run([self.train_op, self.J], feed_dict={self.X: X_prime_batch, self.y: X_batch})
                        if batch_idx % 100 == 0:
                            print('epoch{0}_batch{1}: {2}'.format(epoch, batch_idx, loss))
                            saver.save(sess, ckpt_file)
            
    def add_noise(self, sess, X, corr_frac):
        X_prime = X.copy()
        rand = tf.random_uniform(X.shape)
        X_prime[sess.run(tf.nn.relu(tf.sign(corr_frac - rand))).astype(np.bool)] = 0
        return X_prime
        
    def gen_mini_batches(self, X, batch_size):
        X = np.array(X)
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i + batch_size]
        
        
    def build_model(self):
        print('Build Denoising Autoencoder Model v0.0.8')
        print('begine to build the model')
        self.X = tf.placeholder(shape=[None, self.n], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.n], dtype=tf.float32) 
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.W1 = tf.Variable(
            tf.truncated_normal(
                shape=[self.n, self.hidden_size], mean=0.0, stddev=0.1),
            name='W1')
        self.b2 = tf.Variable(tf.constant(
            0.001, shape=[self.hidden_size]), name='b2')
        self.b3 = tf.Variable(tf.constant(
            0.001, shape=[self.n]), name='b3')
        with tf.name_scope('encoder'):
            z2 = tf.matmul(self.X, self.W1) + self.b2
            self.a2 = tf.nn.tanh(z2)
        with tf.name_scope('decoder'):
            z3 = tf.matmul(self.a2, tf.transpose(self.W1)) + self.b3
            a3 = tf.nn.tanh(z3)
        self.y_ = a3
        r_y_ = tf.clip_by_value(self.y_, 1e-10, float('inf'))
        r_1_y_ = tf.clip_by_value(1 - self.y_, 1e-10, float('inf'))
        cost = - tf.reduce_mean(tf.add(
                tf.multiply(self.y, tf.log(r_y_)),
                tf.multiply(tf.subtract(1.0, self.y), tf.log(r_1_y_))))
        self.J = cost + self.regcoef * tf.nn.l2_loss([self.W1])
        self.train_op = tf.train.AdamOptimizer(0.001,0.9,0.9,1e-08).minimize(self.J)

        
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