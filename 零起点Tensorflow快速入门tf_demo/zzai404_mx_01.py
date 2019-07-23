#coding=utf-8
'''
Created on 2016.12.25
TopQuant-极宽量化系统·培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com

'''

import arrow
import pandas as pd
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
import ztop_ai as zai
import zpd_talib as zta

#
#-----------------------


def mx_fun010(funSgn,x_train, x_test, y_train, y_test,yk0=5,fgInt=False,fgDebug=False):
    #1
    df9=x_test.copy()
    mx_fun=zai.mxfunSgn[funSgn]
    mx =mx_fun(x_train.values,y_train.values)
    #2
    print(df9.tail())
    y_pred = mx.predict(x_test.values)
    print('\n nn')
    print(y_pred[:0].size)
    print(len(df9.index))
    print(len(x_test.index))
    print(len(y_test.index))
    print(len(x_train.index))
    print(len(y_train.index))
    print(y_pred)
    #
    df9['y_test'],df9['y_pred']=y_test,y_pred
    #3   
    if fgInt:
        df9['y_predsr']=df9['y_pred']
        df9['y_pred']=round(df9['y_predsr']).astype(int)
        
    #4
    dacc=zai.ai_acc_xed(df9,yk0,fgDebug)
    #5
    if fgDebug:
        #print(df9.head())
        print('@fun name:',mx_fun.__name__)
        df9.to_csv('tmp/df9_pred.csv');
    #
    #6
    print('@mx:mx_sum,kok:{0:.2f}%'.format(dacc))   
    return dacc,df9   

#-----------------------

#1 
fsr0='data/ccpp_'
print('#1',fsr0)
#x_train, x_test, y_train, y_test=zai.ai_dat_rd(fsr0)
#print(x_train.tail())
#
#========
#   AT,V,AP,RH,PE
df=pd.read_csv('data/ccpp.csv',index_col=False)
#x_test=ai_f_datRd010(fsr,k0=0,fgPr=False):
#df[ysgn]=df[ysgn].astype(float)
#df['y']=round(df['y']*1000).astype(int)
x_test['xat']=df['AT']
y_test['ype']=df['PE']
#
print(df.head())
print(df.tail())

print('type(xs)\n',x_test.tail())

#2
print('\n#2,mx_line')
funSgn='line'
tim0=arrow.now()
print(x_test.tail())
#dacc,df9=mx_fun010(funSgn,x_train, x_test, y_train, y_test,5,False)
dacc,df9=mx_fun010(funSgn,x_test, x_test, y_test, y_test,5,False)
tn=zt.timNSec('',tim0,True)
print(df9.tail())
