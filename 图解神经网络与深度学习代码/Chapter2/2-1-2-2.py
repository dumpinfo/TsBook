# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"一维的ndarray"
array=np.array([1,2,3],np.float32)
#"ndarray转换为tensor"
t=tf.convert_to_tensor(array,tf.float32,name='t')
#"打印张量"
print(t)