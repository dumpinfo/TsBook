# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
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
print('\n#1,set dat')
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.mul(w, x, name='output')
y_ = tf.constant(0.0, name='correct_value')

#2
print('\n#2,set fun')
loss = tf.pow(y - y_, 2, name='loss')
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

#3
print('\n#2,set fun')
for value in [x, w, y, y_, loss]:
    tf.scalar_summary(value.op.name, value)
 
summaries = tf.merge_all_summaries()
 
sess = tf.Session()
summary_writer = tf.train.SummaryWriter('log_simple_stats', sess.graph)
 
sess.run(tf.initialize_all_variables())
for i in range(100):
    summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step)


#1
print('\n#1,set gg')
gg = tf.get_default_graph()

#2
print('\n#2')
op1=gg.get_operations()
print('op,',op1)
print('op,',type(op1))

#3
print('\n#3,in_dat')
in_dat = tf.constant(1.0)
op2=gg.get_operations()
print('in_dat,',in_dat)
print('type(in_dat),',type(in_dat))
print('\nop,',op2)
print('op,',type(op2))
print('\nop.node_def,',op2[0].node_def)

#4
print('\n#4,Session')
ss = tf.Session()
xdat=ss.run(in_dat)
print('xdat',xdat)



