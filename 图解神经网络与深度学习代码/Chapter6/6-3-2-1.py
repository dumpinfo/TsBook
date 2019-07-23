import tensorflow as tf
#"输入张量"
t=tf.constant([2,5,3],tf.float32)
x=tf.log(t)
#"softmax处理"
s=tf.nn.softmax(x,0)
#"创建会话"
session=tf.Session()
#"打印结果"
print(session.run(s))