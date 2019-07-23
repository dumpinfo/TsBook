#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

'''

import tensorflow as tf

#-----------------

#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
x = tf.constant(2.0, name='input')
w = tf.Variable(0.8, name='weight')
y_model = tf.multiply(w, x, name='output')


#2
print('\n#2,set.dat #2')
y_ = tf.constant(0.0, name='correct_value')
loss = tf.pow(y_model - y_, 2, name='loss')

#3
print('\n#3,train_step')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

#4
print('\n#4,summar.misc')
for value in [x, w, y_model, y_, loss]:
    tf.summary.scalar(value.op.name, value)
 
summaries = tf.summary.merge_all()

#5
print('\n#5,Session')
ss = tf.Session()

#6
print('\n#6,summary，rlog',rlog)
xsum= tf.summary.FileWriter(rlog, ss.graph)  

#7
print('\n#7,Session init')
init=tf.global_variables_initializer()
ss.run(init)

#8
print('\n#8,step')
for i in range(100):
    #8.a
    xdat=ss.run(summaries)
    xsum.add_summary(xdat, i)
    x2=ss.run(train_step)
    #------
    #8.b
    x2,y2,w2=ss.run(x),ss.run(y_model),ss.run(w)
    s2=ss.run(loss)
    print(i,'#，y2,loss:',y2,s2,',x2,w2 ',x2,w2 )
    

#9
print('\n#9,close')
ss.close()
xsum.close()
