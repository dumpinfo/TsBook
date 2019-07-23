import numpy as np
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", 'datasets/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'work/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS

g_params = {}