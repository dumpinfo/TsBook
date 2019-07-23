import tensorflow as tf
#"二维张量"
value2d=tf.constant(
        [
         [5,1,4,2],
         [3,9,5,7]
                ],tf.float32
        )
#"求沿1方向上，即每一行的最大值的位置索引"
result=tf.argmax(value2d,axis=1)
#result=tf.arg_max(value2d,1) 该函数已经被舍弃
session=tf.Session()
#"打印结果"
print(session.run(result))