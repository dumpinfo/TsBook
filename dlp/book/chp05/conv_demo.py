import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Conv_Demo:
    def __init__(self):
        pass
        
    def startup(self):
        print('Convolutional Demo')
        img_file = 'datasets/wolfs.jpg'
        img_raw = io.imread(img_file)
        img = img_raw.astype(np.float32) / 255.0
        x = img.reshape([1, 256, 256, 3])
        X = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
        W_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 2], mean=0.0, stddev=0.1))
        b_2 = tf.Variable(tf.zeros([256, 256, 2]))
        z_2 = tf.nn.conv2d(X, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_2
        a_2_relu = tf.nn.relu(z_2)
        a_2_sigmoid = tf.nn.sigmoid(z_2)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            rst_relu = sess.run(a_2_relu, feed_dict={X: x})
            rst_sigmoid = sess.run(a_2_sigmoid, feed_dict={X: x})
            fm1 = rst_relu[0, :, :, 0]
            fm2 = rst_relu[0, :, :, 1]
            fm3 = rst_sigmoid[0, :, :, 0]
            fm4 = rst_sigmoid[0, :, :, 1]
            plt.figure(1)
            plt.subplot(231)
            plt.imshow(img)
            plt.axis('off')
            plt.title('origin')
            plt.subplot(232)
            plt.imshow(fm1)
            plt.axis('off')
            plt.title('ReLU fm1')
            plt.subplot(233)
            plt.imshow(fm2)
            plt.axis('off')
            plt.title('ReLU fm2')
            plt.subplot(235)
            plt.imshow(fm3)
            plt.axis('off')
            plt.title('sigmoid fm1')
            plt.subplot(236)
            plt.imshow(fm4)
            plt.axis('off')
            plt.title('sigmoid fm2')
            plt.show()
