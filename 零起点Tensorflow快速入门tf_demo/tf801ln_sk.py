#coding=utf-8
'''
Created on 2016.12.25
TopQuant-极宽量化系统·培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com

'''


import arrow
import pandas as pd
import matplotlib.pyplot as plt

import sklearn 
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
import zpd_talib as zta
import ztop_ai as zai

#
#-----------------------

#1
print('\n#1,data.set')
fss='data/p100.csv'
df=pd.read_csv(fss,index_col=False)
print(df.tail())

#2
print('\n#2,data.xed')
xs,ys=df['x'].values,df['y'].values
print('type(xs),',type(xs))
xs=xs.reshape(-1, 1) 
df9=df.copy()

#3
print('\n3# 建模')
mx =zai.mx_line(xs,ys)

#4
print('\n4# 预测')
y_pred = mx.predict(xs)
df9['y_pred']=y_pred
df9.to_csv('tmp/df_9.csv',index=False)
print(df9.tail())
  
        
#5
print('\n5# plot')
#5.1
print('\n#5.1,plot,xs,ys')
plt.plot(xs, ys, 'ro', label='sr_data')

#5.2
print('\n#5.2,plot,xs,ypred')
ys2=df9['y_pred'].values
plt.plot(xs, ys2, label='ln_model')
#
#5.3
plt.legend()
plt.show()
#
print('\nok!')
