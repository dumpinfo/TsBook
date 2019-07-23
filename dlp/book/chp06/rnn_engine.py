import numpy as np
import tensorflow as tf

class Rnn_Engine(object):
    def __init__(self):
        self.args = get_args()
        if 'true' == self.args.training:
            self.batch_size = 1
            self.seq_length = 1
        
        
    def train(self):
        build_model()
        
    def build_model(self):
        print('build_model')
        
    def get_args(self):
        parser = argparse.ArgumentParser(
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--data_dir', type=str, default='datasets',
                            help='data directory containing input.txt')
        parser.add_argument('--training', type=str, default='true',
                            help='data directory containing input.txt')
        parser.add_argument('--model', type=str, default='lstm',
                            help='data directory containing input.txt')
        return parser.parse_args()