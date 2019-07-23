import tensorflow as tf
from sda_engine import Sda_Engine

def main(_):
    print('Denoising Autoencoder Project')
    sda_engine = Sda_Engine()
    #sda_engine.train()
    sda_engine.run()
    
    
if '__main__' == __name__:
    tf.app.run()