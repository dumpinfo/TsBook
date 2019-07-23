import tensorflow as tf

slim = tf.contrib.slim
import os
import time

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
height, width = 299, 299


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', stddev=1e-1, name="conv2d"):
    # 定义一个通用的卷积层函数
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        conv = tf.nn.relu(conv)
        return conv


def linear(input_, output_size, stddev=1e-1, bias_start=0.0, relu=True, name='Linear'):
    # 定义一个通用的全连接层
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        fc = tf.nn.xw_plus_b(input_, w, bias, name='fc')
        if relu:
            return tf.nn.relu(fc)
        else:
            return fc


class AlexNet(object):
    def __init__(self, sess, batch_size=64, dataset_name='default', checkpoint_dir=None):
        self.y_dim = 1000
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.frame_size = 3
        self.build_model()
        self.sess = sess

    def build_model(self):
        # 构建模型
        image_dims = [height, width, self.frame_size]
        y_dim = [self.y_dim]

        # 定义网络入口
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.y = tf.placeholder(tf.float32, [self.batch_size] + y_dim, name='y')

        # 定义 forward 和 loss
        self.y_logit = self.alexmodel(self.inputs)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_logit))

        # 定义模型保存
        self.saver = tf.train.Saver()

    def train(self, config):
        # 定义优化器
        en_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        start_time = time.time()

        # 定义计数器
        counter = 0

        # 开始迭代训练
        print('start training...')
        for ae_epoch in range(config.epoch):
            batch_idxs = config.train_size

            for i in range(batch_idxs):
                # 定义输入，这里以随机数代替,读者根据自身需求更改输入
                inputs = tf.random_uniform((self.batch_size, height, width, 3))
                labels = tf.one_hot(tf.constant(1, shape=[self.batch_size]), depth=1000, on_value=1.0, off_value=0.0)
                # 这里需要先从定义的inputs和labels等张量得到具体的数字
                batch_inputs, batch_labels = sess.run([inputs, labels])

                _, loss = self.sess.run([en_optim, self.loss],
                                        feed_dict={self.inputs: batch_inputs, self.y: batch_labels})
                print('InceptionV3_Epoch: [%2d][%4d/%4d] time: %4.4f loss: %.8f ' \
                      % (ae_epoch, i, batch_idxs, time.time() - start_time, loss))
                # 保存模型
                if counter % 10 == 0:
                    self.save(config.checkpoint_dir, counter)
                counter += 1

    def alexmodel(self, inputs):
        # ========================
        # 开始构建alexnet模型
        # ========================
        # 第一层卷积
        conv1 = conv2d(inputs, 64, k_h=11, k_w=11, d_h=4, d_w=4, name='conv1')
        # LRN层和最大池化
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        # 第二层卷积
        conv2 = conv2d(pool1, 192, k_h=5, k_w=5, d_h=1, d_w=1, name='conv2')
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        # 第三层卷积
        conv3 = conv2d(pool2, 384, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3')
        # 第四层卷积
        conv4 = conv2d(conv3, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4')
        # 第五层卷积
        conv5 = conv2d(conv4, 256, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5')
        # 最大池化层
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
        # 三个全连接层
        fc6 = linear(tf.reshape(pool5, [self.batch_size, -1]), 4096, name='fc6')
        fc7 = linear(fc6, 4096, name='fc7')
        fc8 = linear(fc7, 1000, relu=False, name='fc8')

        return fc8

    @property
    def model_dir(self):
        return '{}_{}'.format(self.dataset_name, self.batch_size)

    def save(self, checkpoint_dir, step):
        model_name = 'InceptionV3.model'
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)


# 定义一些超参数
flags = tf.app.flags
flags.DEFINE_integer("epoch", 5, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.1, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 10, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_string("dataset", "XXX.tfrecords", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

if __name__ == '__main__':
    with tf.Session() as sess:
        alexnet = AlexNet(
            sess,
            batch_size=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir)
        alexnet.train(FLAGS)