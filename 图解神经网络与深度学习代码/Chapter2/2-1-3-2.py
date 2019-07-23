# -*- coding: utf-8 -*-
import tensorflow as tf
#"二维张量"
t=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ],
        tf.float32
        )
#"张量的尺寸"
#s=t.shape
s=t.get_shape()
#"打印s及其数据结构类型"
print("'s的值:'")
print(s)
print("'s 的数据结构类型:'")
print(type(s))
print("'s[0]的值:'")
print(s[0])
#"打印s[0]及其数据结构类型"
print("'s[0]的数据结构类型:'")
print(type(s[0]))
#"将 s[0] 转换为整数型"
print("'将s[0]的值转换为整数型:'")
print(s[0].value)
print(type(s[0].value))