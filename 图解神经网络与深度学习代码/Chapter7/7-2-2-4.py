# -*- coding: utf-8 -*-
import tensorflow as tf
#"长度为5的输入张量"
I=tf.constant([3,4,1,5,6],tf.complex64)
#"长度为4的卷积核"
K=tf.constant([1,2,-2,-1],tf.complex64)
#"补0操作"
I_padded=tf.pad(I,[[0,3]])
#"将卷积核翻转180°"
K_rotate180=tf.reverse(K,axis=[0])
#"翻转后进行补0操作"
K_roate180_padded=tf.pad(K_rotate180,[[0,4]])
#"傅里叶变换"
I_padded_fft=tf.fft(I_padded)
#"傅里叶变换"
K_roate180_padded_fft=tf.fft(K_roate180_padded)
#"将以上两个傅里叶变换点乘操作"
Ik_fft=tf.multiply(I_padded_fft,K_roate180_padded_fft)
#"傅里叶逆变换"
Ik=tf.ifft(Ik_fft)
#"因为输入的张量和卷积核都是实数，对以上傅里叶逆变换进行取实部的操纵"
Ik_real=tf.real(Ik)
session=tf.Session()
#"打印结果"
print(session.run(Ik_real))