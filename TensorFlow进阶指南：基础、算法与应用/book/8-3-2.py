import tensorflow as tf

q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=3, dtypes=tf.float32)

with tf.Session() as sess :
    for i in range(0,10): # 10次入队
        sess.run(q.enqueue(i))

    for i in range(0,7): # 7次出队
        print(sess.run(q.dequeue()))