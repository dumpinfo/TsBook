import tensorflow as tf
A = tf.Variable([[1, 2], [3, 4]], dtype = tf.float32, name='A')
B = tf.Variable([[1, 1], [1, 1]], dtype = tf.float32, name='B')
y = tf.matmul(A, B)
z = tf.sigmoid(y)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
   sess.run(init_op)
   z = sess.run(z)
   print(z)
