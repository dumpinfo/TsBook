import os
import sys
import collections
import numpy as np
import tensorflow as tf
from app_global import FLAGS
from app_global import g_params

class Data_Loader(object):
    def __init__(self):
        pass
      
    @staticmethod
    def load_datasets(batch_size, num_steps):
        word_to_id, id_to_word = Data_Loader.load_vocab_dict()
        g_params['word_to_id'] = word_to_id
        g_params['id_to_word'] = id_to_word
        # 载入训练样本集
        X_train, y_train, epoch_size_train = Data_Loader.get_dataset(word_to_id, "ptb.train.txt", batch_size, num_steps)
        # 载入验证样本集
        X_validation, y_validation, epoch_size_validation = Data_Loader.get_dataset(word_to_id, "ptb.valid.txt", batch_size, num_steps)
        # 载入测试样本集
        X_test, y_test, epoch_size_test = Data_Loader.get_dataset(word_to_id, "ptb.test.txt", batch_size, num_steps)
        return X_train, y_train, X_validation, y_validation, X_test, y_test, epoch_size_train, epoch_size_validation, epoch_size_test
    
    @staticmethod
    def get_dataset(word_to_id, filename, batch_size, num_steps, name=None):
        fullname = os.path.join(FLAGS.data_path, filename)
        raw_data = Data_Loader.get_file_word_ids(fullname, word_to_id)
        return Data_Loader.create_dataset(raw_data, batch_size, num_steps)
    
    @staticmethod
    def create_dataset(raw_data, batch_size, num_steps, name=None):
        with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
            t_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
            words_num = tf.size(t_data)
            batch_len = words_num // batch_size
            epoch_size = (batch_len - 1) // num_steps
            epoch_size = tf.identity(epoch_size, name="epoch_size") # 变为tensor类型
            t_data = tf.reshape(t_data[0 : batch_size * batch_len],
                      [batch_size, batch_len])
            batch_idx = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue() # batch_idx
            X = tf.strided_slice(t_data, [0, batch_idx * num_steps],
                         [batch_size, (batch_idx + 1) * num_steps])
            X.set_shape([batch_size, num_steps])
            y = tf.strided_slice(t_data, [0, batch_idx * num_steps + 1],
                         [batch_size, (batch_idx + 1) * num_steps + 1])
            y.set_shape([batch_size, num_steps])
        return X, y, epoch_size
    
    @staticmethod
    def load_vocab_dict():
        # 如果有PKL文件则直接读入
        # 从文件中生成
        filename = os.path.join(FLAGS.data_path, "ptb.train.txt")
        data = Data_Loader.read_file_words(filename)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        id_to_word = dict(zip(range(len(words)), words))
        # 保存到PKL文件
        return word_to_id, id_to_word
    
    @staticmethod
    def read_file_words(filename):
        with tf.gfile.GFile(filename, "r") as f:
            if sys.version_info[0] >= 3:
                #return f.read().replace("\n", "<eos>").split()
                return [chr for chr in f.read()] # 针对汉字情况，英文及数字处理会有问题
            else:
                return f.read().decode("utf-8").replace("\n", "<eos>").split()
    
    @staticmethod
    def get_file_word_ids(filename, word_to_id):
        data = Data_Loader.read_file_words(filename)
        return [word_to_id[word] for word in data if word in word_to_id]
    
    @classmethod
    def cls_method(cls):
        pass