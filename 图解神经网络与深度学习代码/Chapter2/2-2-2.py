# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
#"四维张量:张量中的值满足正态分布（平均值为10,标准差为1）的随机数"
sigma=1
mu=10
result=tf.random_normal([10,4,20,5],mu,sigma,tf.float32)
session=tf.Session()
#"Tensor转换为ndarray"
array=session.run(result)
#"将多维的ndarray转换为一维的ndarray"
array1d=array.reshape([-1])
#""计算并显示直方图""
histogram,bins,patch= plt.hist(array1d,25,facecolor='gray',
                      alpha=0.5,normed=True)
x=np.arange(5,15,0.01)
y=1.0/(math.sqrt(2*np.pi)*sigma)*
  np.exp(-np.power(x-mu,2.0)/(2*math.pow(sigma,2)))
plt.plot(x,y)
plt.show()