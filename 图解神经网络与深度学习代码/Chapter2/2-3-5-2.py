import tensorflow as tf
#"二维张量"
value2d=tf.constant(
        [
        [5,1,4,2],
        [3,9,5,7]
                ],tf.float32
        )
#"创建会话"
session=tf.Session()
#"计算沿0轴方向上的和"
sum0=tf.reduce_sum(value2d,axis=0)
print("沿 0 轴方向上的和:")
print(session.run(sum0))
#"计算沿1轴方向上的和"
sum1=tf.reduce_sum(value2d,axis=1)
print("沿 1 轴方向上的和:")
print(session.run(sum1))
#"计算沿(0,1)平面上的和"
sum01=tf.reduce_sum(value2d,axis=(0,1))
print("沿 (0,1) 平面上的和:")
print(session.run(sum01))