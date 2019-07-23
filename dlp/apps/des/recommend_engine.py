import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from apps.des.app_global import appGlobal as desAg
from model.mf_recommend_engine import MFRecommendEngine as MFRecommendEngine

class RecommendEngine(object):
    def __init__(self):
        print('initialize recommend engine')
        self.n = 2 # 知识点数量
        self.nm = 5 # 题目总数
        self.nu = 5 # 学生总数
        self.lanmeda = 0.1 # L2调整项系数
        self.epochs = 5000 # 训练遍数
        
    def run(self):
        self.Y_ph, self.r, self.mu = self.load_dataset()
        print('y_ph:{0}'.format(self.Y_ph))
        self.train()
        self.predict(3, 4)
        
    def predict(self, ui, xi):
        print(self.Xv[xi])
        Uv = np.transpose(self.UTv)
        print(Uv[ui])
        print(np.dot(self.Xv[xi], Uv[ui]) + self.mu[xi][0])
        
    def load_dataset(self):
        self.n, self.nm, self.nu, ph, r = MFRecommendEngine.load_dataset()
        # 求出mu
        mu = np.zeros(shape=(self.nm, 1))
        for row in range(self.nm):
            sum = 0.0
            num = 0
            for col in range(self.nu):
                if 1 == r[row][col]:
                    sum += ph[row][col]
                    num += 1
            mu[row][0] = sum / num
        print(mu)
        print(ph)
        ph = ph - mu
        print(ph)
        return ph, r, mu
        
    def calDeltaY(self, Y, Y_):
        sum = 0.0
        for row in range(self.nm):
            for col in range(self.nu):
                if 1 == self.r[row][col]:
                    sum += (Y[row][col] - Y_[row][col])*(Y[row][col] - Y_[row][col])
        return sum
    
    def build_model(self):
        print('build model')
        self.Y_ = tf.placeholder(shape=[self.nm, self.nu], dtype=tf.float32, name='Y_')
        self.X = tf.Variable(tf.truncated_normal(shape=[self.nm, self.n], mean=0.0, stddev=0.01, seed=1.0), dtype=tf.float32, name='X')
        self.UT = tf.Variable(tf.truncated_normal(shape=[self.n, self.nu], mean=0.0, stddev=0.01, seed=1.0), dtype=tf.float32, name='X')
        self.Y = tf.matmul(self.X, self.UT)
        self.L = self.calDeltaY(self.Y, self.Y_) #tf.reduce_sum((self.Y - self.Y_)*(self.Y - self.Y_))
        self.J = self.L + self.lanmeda*tf.reduce_sum(self.X**2) + self.lanmeda*tf.reduce_sum(self.UT**2)
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(self.J)
        
    def train(self):
        self.build_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                X, UT, Y, J, train_step = sess.run([self.X, self.UT, self.Y, self.J, self.train_step], feed_dict={self.Y_: self.Y_ph})
                #print(Y)
                print('{0}:{1}'.format(epoch, J))
            self.Xv = X
            self.UTv = UT