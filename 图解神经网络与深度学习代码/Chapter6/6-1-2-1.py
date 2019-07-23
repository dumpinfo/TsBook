# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"读取 tfrecord 文件列表，这里只有一个 tfrecord"
records_queue=tf.train.string_input_producer(['data.tfrecord'],num_epochs=2)
#"创建一个 TFRecordReader 对象"
reader=tf.TFRecordReader()
_,serialized_example=reader.read(records_queue)
#"解析 tfrecord 中的数据,每次只解析一个"
features=tf.parse_single_example(
        serialized_example,
        features={
                'array_raw':tf.FixedLenFeature([],tf.string),
                'height':tf.FixedLenFeature([],tf.int64),
                'width':tf.FixedLenFeature([],tf.int64),
                'depth':tf.FixedLenFeature([],tf.int64)
                }
                                 )
#"解析出对应的值"
array_raw=features['array_raw']
array = tf.decode_raw(array_raw,tf.float32)#"解码"
height= features['height']
width = features['width']
depth = features['depth']
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=session,coord=coord)
#"循环5次解析文件流中的数据"
for i in range(5):
     ndarray,h,w,d=session.run([array,height,width,depth])
     print('---第%(num)d 次解析到的ndarray---'%{'num':i+1})
     print(ndarray)
coord.request_stop()
coord.join(threads)
session.close()