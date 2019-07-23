# -*- coding: utf-8 -*-
import tensorflow as tf
#"输入长度为3的一维张量"
f=tf.constant([4,5,6],tf.complex64)
#"创建会话"
session=tf.Session()
#"一维傅里叶变换"
F=tf.fft(f)
#"打印傅里叶变换F的值"
print("傅里叶变换F的值:")
print(session.run(F))
#"计算F的傅里叶逆变换(显然与输入的f是相等的)"
F_ifft=tf.ifft(F)
#"打印F傅里叶逆变换的值"
print("打印F的傅里叶逆变换的值:")
print(session.run(F_ifft))