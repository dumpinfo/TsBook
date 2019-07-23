import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from app_global import FLAGS
from tensorflow.examples.tutorials.mnist import input_data

class Rbm_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self, name='rbm'):
        self.datasets_dir = 'datasets/'
        self.random_seed = 1 # 用于测试目的，使每次生成的随机数相同
        self.name = name
        #self.loss_func = loss_func
        self.learning_rate = 0.0001
        self.num_epochs = 50
        self.batch_size = 128
        self.regtype = 'l2'
        self.regcoef = 0.00001
        #self.loss = Loss(self.loss_func)
        self.num_hidden = 250
        self.visible_unit_type = 'bin'
        self.gibbs_sampling_steps = 3
        self.stddev = 0.1
        self.W = None
        self.bh_ = None
        self.bv_ = None
        self.w_upd8 = None
        self.bh_upd8 = None
        self.bv_upd8 = None
        self.cost = None
        self.input_data = None
        self.hrand = None
        self.vrand = None
        self.tf_graph = tf.Graph()
        self.n = 784 # 28*28黑白图片
        
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/rbm.ckpt'):
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        with self.tf_graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.num_epochs):
                    np.random.shuffle(X_train)
                    batches = [_ for _ in self.gen_mini_batches(X_train,
                                                    self.batch_size)]
                    total_batches = len(batches)
                    for idx, batch in enumerate(batches):
                        cost_val, spos, _, _, _ = sess.run([self.cost, self.positive, self.w_upd8,
                                self.bh_upd8, self.bv_upd8], feed_dict={
                                self.X: batch, 
                                self.hrand: np.random.rand(batch.shape[0],
                                        self.num_hidden),
                                self.vrand: np.random.rand(batch.shape[0],
                                        batch.shape[1])})
                        if (epoch*total_batches + idx) % 100 == 0:
                            saver.save(sess, ckpt_file)
                        print('{0}_{1}:cost={2} {3}, {4}'.format(epoch, idx, cost_val, type(spos), spos.shape))
        
    def gen_mini_batches(self, X, batch_size):
        X = np.array(X)
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i + batch_size]
        
    def build_model(self):
        print('Build RBM Model')
        self.X = tf.placeholder(shape=[None, self.n], dtype=tf.float32, name='X')
        self.hrand = tf.placeholder(shape=[None, self.num_hidden],
                dtype=tf.float32, name='h')
        self.vrand = tf.placeholder(shape=[None, self.n],
                dtype=tf.float32, name='v')
        self.y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        #
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n, self.num_hidden],
                mean=0.0, stddev=0.1), name='W')
        self.bh_ = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]),
                name='bh')
        self.bv_ = tf.Variable(tf.constant(0.1, shape=[self.n], name='bv'))
        #
        self.encode, _ = self.sample_hidden_from_visible(self.X)
        self.reconstruction = self.sample_visible_from_hidden(
                    self.encode, self.n)
        hprob0, hstate0, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
                    self.X, self.n)
        self.vprob = vprob
        self.positive = self.compute_positive_association(self.X,
                                                     hprob0, hstate0)
        nn_input = vprob
        for step in range(self.gibbs_sampling_steps - 1):
            hprob, hstate, vprob, hprob1, hstate1 = self.gibbs_sampling_step(
                nn_input, self.n)
            nn_input = vprob
        negative = tf.matmul(tf.transpose(vprob), hprob1)
        #
        self.w_upd8 = self.W.assign_add(
            self.learning_rate * (self.positive - negative) / self.batch_size)
        self.bh_upd8 = self.bh_.assign_add(tf.multiply(self.learning_rate,
                tf.reduce_mean(tf.subtract(hprob0, hprob1), 0)))
        self.bv_upd8 = self.bv_.assign_add(tf.multiply(self.learning_rate,
                tf.reduce_mean(tf.subtract(self.X, vprob), 0)))
        clip_inf = tf.clip_by_value(vprob, 1e-10, float('inf'))
        clip_sup = tf.clip_by_value(1 - vprob, 1e-10, float('inf'))
        loss = - tf.reduce_mean(tf.add(
                tf.multiply(self.X, tf.log(clip_inf)),
                tf.multiply(tf.subtract(1.0, self.X),
                tf.log(clip_sup))))
        self.cost = loss + self.regcoef*(tf.nn.l2_loss(self.W) +
                tf.nn.l2_loss(self.bh_) + tf.nn.l2_loss(self.bv_))
        
    def sample_hidden_from_visible(self, vis_layer):
        hprobs = tf.nn.sigmoid(tf.add(tf.matmul(vis_layer, self.W), self.bh_))
        hstates = self.sample_prob(hprobs, self.hrand)
        return hprobs, hstates
        
    def sample_prob(self, probs, rand):
        return tf.nn.relu(tf.sign(probs - rand))
        
    def sample_visible_from_hidden(self, hidden, n_features):
        visible_activation = tf.add(
            tf.matmul(hidden, tf.transpose(self.W)),
            self.bv_
        )
        if self.visible_unit_type == 'bin':
            vprobs = tf.nn.sigmoid(visible_activation)
        elif self.visible_unit_type == 'gauss':
            vprobs = tf.truncated_normal(
                (1, n_features), mean=visible_activation, stddev=self.stddev)
        else:
            vprobs = None
        return vprobs
        
    def gibbs_sampling_step(self, visible, n_features):
        hprobs, hstates = self.sample_hidden_from_visible(visible)
        vprobs = self.sample_visible_from_hidden(hprobs, n_features)
        hprobs1, hstates1 = self.sample_hidden_from_visible(vprobs)
        return hprobs, hstates, vprobs, hprobs1, hstates1
        
    def compute_positive_association(self, visible,
                                     hidden_probs, hidden_states):
        if self.visible_unit_type == 'bin':
            positive = tf.matmul(tf.transpose(visible), hidden_states)
        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(visible), hidden_probs)
        else:
            positive = None
        return positive

        
    def run(self, ckpt_file='work/rbm.ckpt'):
        raw = np.random.normal(0, 0.1, self.n)
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        raw = X_train[3098]
        batch = np.reshape(raw, [1, self.n])
        with self.tf_graph.as_default():
            self.build_model()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, ckpt_file)
                vprob = sess.run(self.vprob, feed_dict={
                                self.X: batch, 
                                self.hrand: np.random.rand(batch.shape[0],
                                        self.num_hidden), 
                                self.vrand: np.random.rand(batch.shape[0],
                                        batch.shape[1])})
                print(vprob)
                plt.figure(1)
                plt.subplot(121)
                img0 = np.reshape(raw, [28, 28])
                plt.imshow(img0, cmap='gray')
                plt.subplot(122)
                img = np.reshape(vprob, [28, 28])
                plt.imshow(img, cmap='gray')
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