import tensorflow as tf
from dae_engine import Dae_Engine

def main(_):
    print('Denoising Autoencoder Project')
    dae_engine = Dae_Engine()
    #dae_engine.train()
    dae_engine.run()
    
    
if '__main__' == __name__:
    tf.app.run()