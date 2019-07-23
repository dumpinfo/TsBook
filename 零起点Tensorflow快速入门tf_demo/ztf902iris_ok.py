#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''


#import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------



#1
print('\n#1,set.dat')



# Load IRIS dataset
#data, labels = load_csv("data/iris4x.csv", categorical_labels=True, n_classes=3)
fss='data/iris2.csv'
df=pd.read_csv(fss)

df['cors']=df['xid']*100
print(df.tail())
#ipd = pd.read_csv("iris.csv")

#===============
cors=['','r','g','b']
for xc in range(1,4):
    css,cor,ksiz='xid_'+str(xc),cors[xc],xc*10
    df2=df[df['xid']==xc]
    if xc==1:
        ax = df2.plot.scatter(x='x1', y='x2', color=cor,label=css, s=ksiz);
    else:
        df2.plot.scatter(x='x1', y='x2',  color=cor, label=css, s=ksiz, ax=ax);

#df.plot()
#for xc in range(1,4):
    #df2=df[df['xid']==xc]
    #df2.plot(x='x1', y='x2', kind='scatter',color='xid')
#df.plot(x='x1', y='x2', kind='scatter',color='xid')   
#df.plot.scatter(x='x1', y='x2', color='c')    
'''
plt.subplot(2,1,1)
for key,val in df.groupby('xid'):
    print(key,'v',val)
    
plt.plot(df['x1'], df['x2'], label=key, linestyle="_",  marker='.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.subplot(2,1,2)
#for key,val in df.groupby('xid'):
plt.plot(df['x3'], df['x4'], label=key, linestyle="_",  marker='.')
plt.xlabel('x3')
plt.ylabel('x4')   
plt.legend(loc='best')
plt.show()
'''