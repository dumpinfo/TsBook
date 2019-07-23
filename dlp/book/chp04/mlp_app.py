import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mlp_engine import Mlp_Engine

class Mlp_App:
    def __init__(self):
        pass
        
    def startup(self):
        print('Multi Layer Percepton Startup')
        mlp_engine = Mlp_Engine('datasets')
        #mlp_engine.load_datasets()
        #mlp_engine.train()
        mlp_engine.run()

def main(_):
    mlp_app = Mlp_App()
    mlp_app.startup()
        
if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)