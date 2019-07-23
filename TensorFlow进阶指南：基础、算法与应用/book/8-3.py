import tensorflow as tf

q = tf.FIFOQueue(3, tf.float32)
init = q.enqueue_many(([4, 5, 6],))
x = q.dequeue()

with tf.Session() as sess:
    sess.run(init)
    for j in range(sess.run(q.size())):
        print(sess.run(q.dequeue()))