import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from conv_demo import Conv_Demo
from cnn_engine import Cnn_Engine

class Cnn_App:
    def __init__(self):
        pass
        
    def startup(self):
        print('Convolutional Neural Network Startup')
        #conv_demo = Conv_Demo()
        #conv_demo.startup()
        cnn_engine = Cnn_Engine('datasets')
        #cnn_engine.build_model()
        #cnn_engine.train()
        cnn_engine.run()

def main(_):
    cnn_app = Cnn_App()
    cnn_app.startup()
        
if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)