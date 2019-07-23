import inspect
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from app_global import FLAGS
from tensorflow.examples.tutorials.mnist import input_data
from dae_engine import Dae_Engine
from mlp_engine import Mlp_Engine

class Sda_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self):
        self.datasets_dir = 'datasets/'
        self.random_seed = 1 
        self.dae_W = []
        self.dae_b = []
        self.daes = []
        self.dae_graphs = []
        self.layers = [1024,784,512,256]
        self.name = 'sda'
        prev = 784
        self.mlp_engine = None
        for idx, layer in enumerate(self.layers):
            dae_str = 'dae_' + str(idx+1)
            name = self.name + '_' + dae_str
            tf_graph = tf.Graph()
            self.daes.append(Dae_Engine(name, tf_graph=tf_graph, n=prev, 
                            hidden_size=layer))
            prev = layer
            self.dae_graphs.append(tf_graph)
        
    def run(self, ckpt_file='work/dae.ckpt'):
        if self.mlp_engine is None:
            self.mlp_engine = Mlp_Engine(self.daes, 'datasets')
        self.mlp_engine.run()
        
    def pretrain(self, X_train, X_validation):
        X_train_prev = X_train
        X_validation_prev = X_validation
        for idx, dae in enumerate(self.daes):
            print('pretrain:{0}'.format(dae.name))
            tf_graph = self.dae_graphs[idx]
            X_train_prev, X_validation_prev = self.pretrain_dae(
                            self.dae_graphs[idx], dae, 
                            X_train_prev, X_validation_prev)
        return X_train_prev, X_validation_prev
        
    def pretrain_dae(self, graph, dae, X_train, X_validation):
        dae.train(X_train, X_validation)
        X_train_next = dae.transform(graph, X_train)
        X_validation_next = dae.transform(graph, X_validation)
        return X_train_next, X_validation_next
        
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/dae.ckpt'):
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        self.pretrain(X_train, X_validation)
        if self.mlp_engine is None:
            self.mlp_engine = Mlp_Engine(self.daes, 'datasets')
        self.mlp_engine.train()
        
    def build_model(self):
        print('Build stack denoising autoencoder')
        
    def load_datasets(self):
        mnist = input_data.read_data_sets(self.datasets_dir, 
                one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist