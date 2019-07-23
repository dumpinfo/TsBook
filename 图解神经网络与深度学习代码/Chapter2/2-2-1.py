# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
#"四维张量:张量中的值满足最小值0，最大值10的均匀分布的随机数"
x=tf.random_uniform([10,4,20,5],minval=0,maxval=10,dtype=tf.float32)
session=tf.Session()
#"Tensor转换为ndarray"
array=session.run(x)
#"为了画出直方图,将array转为1个一维的ndarray"
array1d=array.reshape([-1])
plt.hist(array1d)
plt.show()