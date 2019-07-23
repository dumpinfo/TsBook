# -*- coding: utf-8 -*-
import numpy as np
f=np.array([
           [10,2,8],
           [5,12,3]
           ],np.complex64)
#"第一步：对每一列进行傅里叶变换"
f_0_fft=np.fft.fft(f,axis=0)
print(f_0_fft)
#"第二步：针对第一步得到的结果，分别对每一行进行傅里叶变换"
f_0_1_fft=np.fft.fft(f_0_fft,axis=1)
print(f_0_1_fft)