#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

pkmital案例合集

@from:
pkmital案例合集网址：
https://github.com/pkmital/tensorflow_tutorials

'''

import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import data

#----------------------------

#17.1
rlog='/ailib/log_tmp'
print('\n#17.1，定义一个通用的 gabor 伽柏函数：')
def gabor(n_values=32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                       (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
    gauss_kernel = tf.matmul(
        tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.multiply(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel

#17.2
print('\n#17.2，调用gabor 伽柏函数，并用图像显示计算数值:')
ss = tf.Session()

z_gabor2=ss.run(gabor())
plt.imshow(z_gabor2)
plt.show()
xsum= tf.summary.FileWriter(rlog, ss.graph) 

#17.3
print('\n#17.3，创建一个convolve卷积函数:')
def convolve(img, W):
    #17.3.a
    # 输入参数W，是一个2D二维数组
    # conv2d二维卷积函数需要一个4d四维的tensor张量数据，其格式为:
    #[高height,宽width,输入数据数目n_input,输出入数据数目]
    
    #17.3.b  
    if len(W.get_shape()) == 2:
        dims = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)
        
    #17.3.c  
    if len(img.get_shape()) == 2:
        #如果图像是2通道
        #[数目num,高度height,宽度width,通道数]
        dims = [1] + img.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)
        
    #17.3.d    
    elif len(img.get_shape()) == 3:
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        #如果图像是3通道
        #卷积核函数需要为每个通道进行重复concat连接计算
        W = tf.concat(axis=2, values=[W, W, W])

    #17.3.e
    # 计算conv2d二维卷积函数
    # 计算时，需要跳过变量数值由以下数据决定
    #[数目num,高度height,宽度width,通道数]
    convolved = tf.nn.conv2d(img, W,strides=[1, 1, 1, 1], padding='SAME')
    
    #17.3.f
    return convolved

#17.4
print('\n#17.4，调入一个宇航员的图像，并显示:')
img = data.astronaut()
plt.imshow(img)
plt.show()
print(img.shape)

#17.5
print('\n#17.5，创建占位符，用于输入数据:')
x = tf.placeholder(tf.float32, shape=img.shape)

#17.6
print('\n#17.6，用gabor 伽柏函数，作为卷积函数的输入参数，建立graph图运算模型:')
out = convolve(x, gabor())

#17.7
print('\n#17.7，img图像作为输入数据，传递给graph计算模型out，并用图像，显示计算结果:')
#result = tf.squeeze(out).eval(feed_dict={x: img})
result = ss.run(tf.squeeze(out),feed_dict={x: img})
plt.imshow(result)
plt.show()
