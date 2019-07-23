# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
f=tf.constant([
           [10,2,8],
           [5,12,3]
           ],tf.complex64)
#"创建会话"
session=tf.Session()
#"f的二维离散傅里叶变换"
F=tf.fft2d(f)
#"打印f的傅里叶变换的值"
print("f的二维离散傅里叶变换:")
print(session.run(F))
#"计算F的傅里叶逆变换(显然与输入的f是相等的)"
F_ifft2d=tf.ifft2d(F)
#"打印F的傅里叶逆变换的值"
print("F的傅里叶逆变换:")
print(session.run(F_ifft2d))