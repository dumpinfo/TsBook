import numpy as np
import tensorflow as tf

def main():
    A = tf.fill([2, 3], 3.0)
    B = tf.fill([2, 3], 1.0)
    C = tf.fill([3, 2], 5.0)
    I = tf.diag([1.0, 1.0, 1.0])
    with tf.Session() as sess:
        rst_A = sess.run(A)
        rst_A_p_B = sess.run(A+B)
        print('矩阵加法：\r\n{0}'.format(rst_A_p_B))
        rst_A_s_B = sess.run(A-B)
        print('矩阵减法：\r\n{0}'.format(rst_A_s_B))
        rst_A_m_C = sess.run(tf.matmul(A, C))
        print('矩阵乘注：\r\n{0}'.format(rst_A_m_C))
        rst_I = sess.run(I)
        print('单位矩阵：\r\n{0}'.format(rst_I))
        rst_A_m_I = sess.run(tf.matmul(A, I))
        print('矩阵乘以单位矩阵：\r\n{0}'.format(rst_A_m_I))
                
if '__main__' == __name__:
    main()