# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
#"读取图片文件"
image=tf.read_file("test.jpg",'r')
#"将图片文件解码为Tensor"
image_tensor=tf.image.decode_jpeg(image)
#"图像张量的形状"
shape=tf.shape(image_tensor)
session=tf.Session()
print('图像的形状:')
print(session.run(shape))
#"Tensor 转换为 ndarray"
image_ndarray=image_tensor.eval(session=session)
#"显示图片"
plt.imshow(image_ndarray)
plt.show()