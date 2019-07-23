# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


#%%
def read_and_decode(filename, batch_size): # read train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [208, 208, 3])  #reshape image to 512*80*3
#    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    label = tf.cast(features['label'], tf.int32) #throw label tensor

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size= batch_size,
                                                    num_threads=64,
                                                    capacity=2000,
                                                    min_after_dequeue=1500,
                                                    )
    return img_batch, tf.reshape(label_batch,[batch_size])

#%%
tfrecords_file = 'dog_train.tfrecords'
BATCH_SIZE = 20
image_batch, label_batch = read_and_decode(tfrecords_file, BATCH_SIZE)

with tf.Session()  as sess:

    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<1:
            # just plot one batch size
            image, label = sess.run([image_batch, label_batch])
            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(image[j,:,:,:])
                plt.show()
            i+=1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()