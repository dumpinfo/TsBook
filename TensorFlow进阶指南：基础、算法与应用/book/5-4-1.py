import tensorflow as tf
def inference(inputs):
  # input shape: [batch, height, width, 1]
  with tf.variable_scope('conv1'):
    weights = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
    biases = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
  maxpool2 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
  with tf.variable_scope('conv3'):
    weights = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
    biases = tf.Variable(tf.zeros([16]))
    conv3 = tf.nn.conv2d(maxpool2, weights, strides=[1, 1, 1, 1])
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))
  maxpool4 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
  with tf.variable_scope('conv5'):
    weights = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
    biases = tf.Variable(tf.zeros([16]))
    conv5 = tf.nn.conv2d(maxpool4, weights, strides=[1, 1, 1, 1])
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases))
  with tf.variable_scope('fc6'):
    flat = tf.reshape(conv5, [-1, 120])
    weights = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
    biases = tf.Variable(tf.zeros([84]))
    fc6 = tf.nn.matmul(flat, weights) + biases
    fc6 = tf.nn.relu(fc6)
  with tf.variable_scope('fc7'):
    weights = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
    biases = tf.Variable(tf.zeros([10]))
    fc7 = tf.nn.matmul(fc6, weights) + biases
    fc7 = tf.nn.softmax(fc7)
  return fc7