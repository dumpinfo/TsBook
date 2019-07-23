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

#----------------------------
#1
print('\n#1，第一个TF张量')
rlog='/ailib/log_tmp'
n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)
print('x,',x)


#2
print('\n#2，构建tf.Session，执行图运算。')
sess = tf.Session()
xsum= tf.summary.FileWriter(rlog, sess.graph) 
result = sess.run(x)
print('result,',result)

#3
print('\n#3，通过eval表达式函数执行session会话操作')
print('x.eval() 空函数不会工作，需要一个session会话变量作为参数')
xss=x.eval(session=sess)
print('x.eval(session=sess),',xss)

#4
print('\n#4，无需保存会话过程时，可以设置为InteractiveSession互动会话模式')
sess.close()
sess = tf.InteractiveSession()

#5
print('\n#5，InteractiveSession互动会话模式下')
print('x.eval() 空函数可以工作，无需输入参数')
# %现在这个工作！
xss=x.eval()
print('x.eval(),',xss)

#6
print('\n#6，学习tf.Operation操作')
print('使用 [-3, 3]范围内的变量，创造一个高斯分布')
sigma = 1.0
mean = 0.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
print('z,',z)
print('z,',z.eval())

#7
print('\n#7，默认情况下，新的Ops操作添加到默认图')
print('断言assert 用z.graph作为默认图TF.get_default_graph()的别名。')
assert z.graph is tf.get_default_graph()
print('z.graph,',z.graph)

#8
print('\n#8，执行图运算，并用结果数据绘图')
zdat=z.eval()
print('zss,',zdat)
plt.plot(zdat)
plt.show()

#9
print('\n#9，张量z的形状shape数值：')
zdat=z.get_shape()
print('z.get_shape(),',zdat)

#10
print('\n#10，张量z的形状shape数值，采用更友好的list列表显示模式。')
zlst=z.get_shape().as_list()
print('z.get_shape().as_list(),',zlst)

#11
print('\n#11，有时，在 graph 图计算完成前，我们无法知道张量形状（shape of a tensor）')
print('在这种情况下，应该用tf.shape函数，获取张量tensor最终的形状数值shape，')
print('而不是一组tf.Dimension维度数据的离散数据。')
zdat=tf.shape(z).eval()
print('print(tf.shape(z).eval()),',zdat)

#12
print('\n#12，我们可以合并计算张量数据：')
zdat=tf.stack([tf.shape(z), tf.shape(z), [3], [4]]).eval()
print('tf.stack([tf.shape(z), tf.shape(z), [3], [4]]).eval(),\n',zdat)

#13
print('\n#13，两个张量相乘，生成一个2d二维高斯矩阵：')
z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
print('tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values])),',z_2d)

#14
print('\n#14，计算2d二维高斯矩阵的数值，并用图像显示：')
z_2dx=z_2d.eval()
plt.imshow(z_2dx)
plt.show()

#15
print('\n#15，建立一个 gabor 伽柏函数补丁：')
x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.multiply(tf.matmul(x, y), z_2d)
z_gabor=z.eval()
plt.imshow(z_gabor)
plt.show()

#16
print('\n#16，列出一个图graph的所有操作Ops：')
ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])

