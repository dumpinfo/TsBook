import tensorflow as tf
from dbn_engine import Dbn_Engine

def main(_):
    print('Denoising Autoencoder Project')
    dbn_engine = Dbn_Engine()
    #dbn_engine.train()
    dbn_engine.run()
    
    
if '__main__' == __name__:
    tf.app.run()