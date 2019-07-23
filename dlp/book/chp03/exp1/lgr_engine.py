import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from seg_ds_loader import Seg_Ds_Loader

class Lgr_Engine(object):
    def __init__(self, datasets_file, test_file, n, c, batch_size):
        self.datasets_file = datasets_file #'datasets/linear_data_train.csv'
        self.train_file = 'datasets/train.csv'
        self.test_file = test_file #'datasets/linear_data_eval.csv'
        self.n = n # 特征向量维度
        self.c = c # 类别数
        self.batch_size = batch_size
        
    def train(self, mode=0):
        ''' mode=0为全新训练；mode=1为继续训练 '''
        X_train, y_train, X_validation, y_validation, X_test, y_test, m = self.load_datasets()
        X, y_, W, b, y, cross_entropy, train_step, predicted_class, correct_prediction, accuracy = self.build_model()
        # 装入测试样本
        test_data_node = tf.constant(X_test)
        num_epochs = 5
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                batch_num = m // self.batch_size
                for batch_idx in range(batch_num):
                    offset = batch_idx * self.batch_size
                    X_mb = X_train[offset:(offset+self.batch_size), :]
                    y_mb = y_train[offset:(offset+self.batch_size)]
                    # train_step.run(feed_dict={X: X_mb, y_: y_mb})
                    sess.run(train_step, feed_dict={X: X_mb, y_: y_mb})
            print('W:{0}'.format(sess.run(W)))
            print('b:{0}'.format(sess.run(b)))
            print("Accuracy:", accuracy.eval(feed_dict={X: X_test, y_: y_test}))
            saver.save(sess, 'work/lgr.ckpt')
            self.plot(sess.run(W), sess.run(b), self.train_file)
            
    def run(self, x):
        print('run in run mode')
        X, y_, W, b, y, cross_entropy, train_step, predicted_class, \
                correct_prediction, accuracy = self.build_model()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, 'work/lgr.ckpt')
            y_val = sess.run(y, feed_dict={X: x})
            print('rst:{0} \r\nc1:{1} c2:{2}'.format(y_val, 
                    y_val[0][0], y_val[0][1]))
        
    def load_datasets(self):
        loader = Seg_Ds_Loader()
        train_file, validation_file, test_file = loader.prepare_datesets(
                self.datasets_file, self.test_file)
        num_labels = 2
        X_train, y_train = loader.load_dataset(
                train_file, num_labels)
        X_validation, y_validation = loader.load_dataset(
                validation_file, num_labels)
        X_test, y_test = loader.load_dataset(
                test_file, num_labels)
        return X_train, y_train, X_validation, \
                y_validation, X_test, y_test, X_train.shape[0]
        
    def build_model(self):
        X = tf.placeholder("float", shape=[None, self.n])
        y_ = tf.placeholder("float", shape=[None, self.c])
        W = tf.Variable(tf.zeros([self.n, self.c]))
        b = tf.Variable(tf.zeros([self.c]))
        z = tf.matmul(X,W) + b
        y = tf.nn.softmax(tf.matmul(X,W) + b)
        cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # ??效果好，仅需5次迭代
        #cross_entropy = tf.reduce_sum(tf.pow(y_-y, 2)) / (2*m) # 需要10000次左右迭代才能得到有意义结果
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        # 评估模型
        predicted_class = tf.argmax(y, 1);
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return X, y_, W, b, y, cross_entropy, train_step, predicted_class, correct_prediction, accuracy;
        
        
    def plot(self, W, b, filename):
        x01 = []
        x02 = []
        x11 = []
        x12 = []
        ds_reader = csv.reader(open(filename, encoding='utf-8'))
        for row in ds_reader:
            if int(row[0]) > 0:
                x11.append(float(row[1]))
                x12.append(float(row[2]))
            else:
                x01.append(float(row[1]))
                x02.append(float(row[2]))
        plt.scatter(x01, x02, s=20, color='r')
        plt.scatter(x11, x12, s=20, color='b')
        
        w1 = W[0][0]
        w2 = W[1][0]
        b_ = b[0]
        x1 = np.linspace(-0.2, 1.2, 100)
        x2 = [-(w1/w2)*x-b_/w2 for x in x1]
        plt.plot(x1, x2, 'g-', label='Plan', linewidth=2)
        
        plt.show()