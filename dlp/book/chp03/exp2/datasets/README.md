import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from lgr_engine import Lgr_Engine

class Lgr_App:
    def __init__(self):
        pass
        
    def startup(self):
        print('Logistic Regression Startup')
        #lgr_engine = Lgr_Engine(FLAGS.data_dir)
        #lgr_engine.train(mode=Lgr_Engine.TRAIN_MODE_CONTINUE)
        #lgr_engine.run()
        self.learn_mnist()
        
    def test(self):
        img_file = 'datasets/test1.png'
        #img = mpimg.imread(img_file)
        img=io.imread(img_file, as_grey=True)
        print('io: shape:{0}'.format(img.shape))
        raw = img.reshape(784)
        fine = [1 if x<0.5 else 0 for x in raw]
        fine1 = np.array(fine)
        img2 = fine1.reshape(28, 28)
        plt.imshow(img2, cmap='gray')
        plt.title('result:')
        plt.axis('off')
        plt.show()
        print('X: {0}'.format(img2))
        
    def learn_mnist(self):
        mnist = input_data.read_data_sets(FLAGS.data_dir, 
                one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        print('X_train: {0} y_train:{1}'.format(
                X_train.shape, y_train.shape))
        print('X_validation: {0} y_validation:{1}'.format(
                X_validation.shape, y_validation.shape))
        print('X_test: {0} y_test:{1}'.format(
                X_test.shape, y_test.shape))
        image_raw = (X_train[1] * 255).astype(int)
        image = image_raw.reshape(28, 28)
        label = y_train[1]
        idx = 0
        for item in label:
            if 1 == item:
                break
            idx += 1
        plt.title('digit:{0}'.format(idx))
        plt.imshow(image, cmap='gray')
        plt.show()

def main(_):
    lgr_app = Lgr_App()
    lgr_app.startup()
        
if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)