import tensorflow as tf
t1=tf.constant(
        [
        [1,2,3],
        [4,5,6]
        ],tf.float32
        )
t2=tf.constant(
        [
        [11,12],
        [24,25]
        ],tf.float32
        )
#"两个张量沿1方向上连接"
t=tf.concat([t1,t2],axis=1)
session=tf.Session()
print(session.run(t))