import tensorflow as tf
from rbm_engine import Rbm_Engine

def main(_):
    print('Denoising Autoencoder Project')
    rbm_engine = Rbm_Engine()
    #rbm_engine.train()
    rbm_engine.run()
    
    
if '__main__' == __name__:
    tf.app.run()