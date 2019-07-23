import numpy as np
import tensorflow as tf

def main():
    A = tf.constant(value=[[1, -3.0, 3.0], [3, -5, 3], 
                    [6, -5, 4]], dtype=tf.float32)
    e, v = tf.self_adjoint_eig(A)
    with tf.Session() as sess:
        rst_A = sess.run(A)
        rst_e, rst_v = sess.run([e, v])
        print('A:\r\n{0}'.format(rst_A))
        print('特征值：{0}'.format(rst_e))
        print('特征向量：\r\n{0}'.format(rst_v))
        
                
if '__main__' == __name__:
    main()