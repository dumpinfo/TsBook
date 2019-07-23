import tensorflow as tf
with tf.name_scope("my_scope"):
    a = tf.get_variable("value1", [1], dtype=tf.float32)
    b = tf.Variable(1, name="value2", dtype=tf.float32)
    c = tf.add(a, b)
print(a.name)
print(b.name)
print(c.name)
print("------------华丽的分割线-------------")
with tf.variable_scope("my_scope"):
    a = tf.get_variable("value1", [1], dtype=tf.float32)
    b = tf.Variable(1, name="value2", dtype=tf.float32)
    c = tf.add(a, b)
print(a.name)
print(b.name)
print(c.name)