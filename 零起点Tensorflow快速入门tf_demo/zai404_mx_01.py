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
    y_pred = mx.predict(x_test.values)
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
x_train, x_test, y_train, y_test=zai.ai_dat_rd(fsr0)
print(x_train.tail())
print(y_train.tail())

#2
print('\n#2,mx_line')
funSgn='line'
tim0=arrow.now()
dacc,df9=mx_fun010(funSgn,x_train, x_test, y_train, y_test,5,False)
tn=zt.timNSec('',tim0,True)
    
#3
print('\n#3,mx_log')
funSgn='log'
tim0=arrow.now()
print(x_test.tail())
dacc,df9=mx_fun010(funSgn,x_train, x_test, y_train, y_test,5,False,True)
tn=zt.timNSec('',tim0,True)

  
#-----------------------    
print('\nok!')


#==========
'''

#2,mx_line
         AT      V       AP     RH      x
2387  26.34  69.45  1013.87  54.15  25.97
2388  25.06  64.63  1020.66  54.93   6.86
2389   8.34  40.96  1023.28  89.45  18.28
2390  26.69  70.36  1006.82  68.29  15.33
2391  17.46  53.29  1018.07  91.01  10.33
Traceback (most recent call last):
    
'''