# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#"创建文件"
record=tf.python_io.TFRecordWriter('dataTest.tfrecord')
#"高1宽2深度3的三维ndarray"
array1=np.array(
        [
        [[1,2,3],[4,5,6]],
        ],np.float32
        )
#"高1宽2深度3的三维ndarray"
array2=np.array(
        [
        [[11,12,13],[14,15,16]],
        ],np.float32
        )
#"高1宽2深度3的三维ndarray"
array3=np.array(
        [
        [[21,23,21],[23,24,22]],
        ],np.float32
        )
#"将上述的3个ndarray存入一个列表"
arrays=[array1,array2,array3]
#"循环处理上述列表中的每一个ndarray"
for array in arrays:
    #"将ndarray中的值转为为字节类型"
    array_raw=array.tostring()
    #"ndarray的值"
    feature={
         'array_raw':
             tf.train.Feature(bytes_list=tf.train.
                   BytesList(value=[array_raw])),
         }
    features=tf.train.Features(feature=feature)
    example=tf.train.Example(features=features)
    #"字符串序列化后写入文件"
    record.write(example.SerializeToString())
record.close()