import numpy as np
import tensorflow as tf
from lstm_engine import Lstm_Engine

def main(_):
    print('LSTM Project')
    lstm_engine = Lstm_Engine()
    #lstm_engine.train()
    lstm_engine.run()
    
def test():
    lstm_size = 28
    time_step_size = 28
    raw = np.ndarray(shape=(128, 28, 28), dtype=int)
    for i in range(128):
        for j in range(28):
            for k in range(28):
                raw[i][j][k] = 10000000 + (i+1) * 10000 + (j+1)*100 + (k+1)
    X = tf.placeholder("float", [None, 28, 28])
    XT = tf.transpose(X, [1, 0, 2])
    XR = tf.reshape(XT, [-1, lstm_size])
    X_split = tf.split(XR, time_step_size, 0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        rst = sess.run(X_split, feed_dict={X: raw})
        rst = np.array(rst)
        print('shape={0}, data={1}'.format(rst.shape, rst))
if '__main__' == __name__:
    tf.app.run()
    #test()