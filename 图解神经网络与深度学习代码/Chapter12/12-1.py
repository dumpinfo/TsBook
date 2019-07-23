# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"x是1个3行3列深度1的张量"
x=tf.placeholder(tf.float32,(1,3,3,1))
#"2x2的掩码,步长是(1,1,1,1)的VALID平均值池化操作"
sigma=tf.nn.avg_pool(x,(1,2,2,1),(1,1,1,1),'VALID')
#"利用上述池化操作的结果，构造一个函数F:池化结果的和"
F=tf.reduce_sum(sigma)
#"创建会话"
session=tf.Session()
#"分别计算 F 在某一点 xvalue 处 针对 sigma 和 x 的梯度"
xvalue=np.random.randn(1,3,3,1)
grad=tf.gradients(F,[sigma,x])
results=session.run(grad,{x:xvalue})
#"打印结果"
print("----针对sigma的梯度----:")
print(results[0])
print("----针对x的梯度----:")
print(results[1])