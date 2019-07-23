import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from app_global import appGlobal as ag
from recommend_engine import RecommendEngine

def main(_):
    print('Education Recommend System:{0} v0.0.1'.format(ag.FLAGS.epoch))
    recommendEngine = RecommendEngine()
    recommendEngine.run()
    
if '__main__' == __name__:
    tf.app.run()