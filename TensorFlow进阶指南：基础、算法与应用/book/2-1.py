import tensorflow as tf
hello = tf.constant('hello Tensorflow!')
sess = tf.Session()
print(sess.run(hello))
