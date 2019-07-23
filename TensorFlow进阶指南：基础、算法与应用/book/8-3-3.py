import tensorflow as tf

q = tf.FIFOQueue(10, "float")
counter = tf.Variable(0.0)  #计数器
# 给计数器加一
increment_op = tf.assign_add(counter, 1.0)
# 将计数器加入队列
enqueue_op = q.enqueue(counter)

qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 2)

# 主线程
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 启动入队线程
qr.create_threads(sess, start=True)
for i in range(20):
    print(sess.run(q.dequeue()))