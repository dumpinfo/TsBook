# -*- coding: utf-8 -*-
import tensorflow as tf
#"创建TFRecordReader对象"
epochs=2
reader=tf.TFRecordReader()
records_queue=tf.train.string_input_producer(['dataTest.tfrecord'],
                                              num_epochs=epochs)
_,serialized_example=reader.read(records_queue)
#"解析文件中的图像及其对应的标签"
features=tf.parse_single_example(
        serialized_example,
        features={
                'array_raw':tf.FixedLenFeature([],tf.string)
                }
                                 )
#"解码二进制数据"
array_raw=features['array_raw']
array_raw=tf.decode_raw(array_raw,tf.float32)
array=tf.reshape(array_raw,[1,2,3])
#"每次从文件中读取2个数据"
BatchSize =2#"不能大于文件中数据的个数"
arrays=tf.train.shuffle_batch([array],BatchSize,1000+3*BatchSize,1000)
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=session,coord=coord)
#"循环2次，从文件中随机读取"
for e in range(2):
    arrs=session.run([arrays])
    print('---第%(num)d批array---'%{'num':e+1})
    print(arrs)
coord.request_stop()
coord.join(threads)
session.close()