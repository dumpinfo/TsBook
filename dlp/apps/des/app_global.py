import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")

class AppGlobal(object):
    def __init__(self):
        self.FLAGS = flags.FLAGS
        

appGlobal = AppGlobal()