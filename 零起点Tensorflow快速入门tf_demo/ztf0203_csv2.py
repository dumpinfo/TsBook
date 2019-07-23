# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


IRIS_TRAINING = "./iris_training.csv"
IRIS_TEST = "./iris_test.csv"

# 数据集读取，训练集和测试集
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)

