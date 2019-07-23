# -*- coding: utf-8 -*-
import tensorflow as tf
#"构建全连接网络"
def net(tensor):
    #""输入层、隐含层、输出层的神经元个数""
    I,H1,O=784,200,10
    #"第1层的权重矩阵和偏置"
    w1=tf.random_normal([I,H1],0,1,tf.float32)
    b1=tf.random_normal([H1],0,1,tf.float32)
    #"隐含层的结果，采用 sigmoid 激活函数"
    l1=tf.matmul(tensor,w1)+b1
    sigma1=tf.nn.sigmoid(l1)
    #"第2层的权重矩阵和偏置"
    w2=tf.random_normal([H1,O],0,1,tf.float32)
    b2=tf.random_normal([O],0,1,tf.float32)
    #"输出层的结果"
    l2=tf.matmul(sigma1,w2)+b2
    return l2
#"读取图片文件"
image=tf.read_file("0.jpg",'r')
#"将图片文件解码为Tensor"
image_tensor=tf.image.decode_jpeg(image)
#"图像张量的形状"
length=tf.size(image_tensor)#length=28*28
#"改变形状，拉伸为一个一维张量，按行存储"
t=tf.reshape(image_tensor,[1,length])
#"数据类型转换，转换为float32类型"
t=tf.cast(t,tf.float32)
#"标准化处理"
t=t/255.0
#"将其输入定义的2层全连接网络"
output=net(t)
session=tf.Session()
#打印结果：
print(session.run(output))