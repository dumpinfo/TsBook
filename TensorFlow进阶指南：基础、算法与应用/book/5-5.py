import tensorflow as tf

slim = tf.contrib.slim
import os
import time

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
height, width = 299, 299


class InceptionV3(object):
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
        self.y_logit, self.end_points = self.inception_v3(self.inputs, is_training=True)
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

    def inception_v3_arg_scope(self, weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
        # ===============================
        # 该函数用来生成网络中经常用到的函数的默认参数，如卷积的激活函数、权重初始化方式、标准化器等。
        # ===============================
        batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': {
                'beta': None,
                'gamma': None,
                'moving_mean': [batch_norm_var_collection],
                'moving_variance': [batch_norm_var_collection]
            }
        }
        # slim.arg_scope是一个非常有用的工具，可以给函数的参数自动赋予某些默认值，使得后面定义卷积层变得非常方便。
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    normalizer_params=batch_norm_params) as sc:
                return sc

    def inception_v3_base(self, inputs, scope=None):
        # ========================================
        # 该函数用于生成Inception V3网络的卷积部分
        # ========================================

        # 定义一个字典end_points用于保存某些关键节点供之后使用
        end_points = {}
        with tf.variable_scope(scope, 'InceptionV3', [inputs]) as scope:
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
                net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
                net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
                net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
                net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
            # 接下来是连续三个Inception模块组，这三个模块组各自分别有多个Inception Module.
            # Inception block
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                # first group inceptions
                with tf.variable_scope('Mixed_5b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                with tf.variable_scope('Mixed_5c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                with tf.variable_scope('Mixed_5d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                # second group inceptions
                with tf.variable_scope('Mixed_6a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], 3)

                with tf.variable_scope('Mixed_6b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                with tf.variable_scope('Mixed_6c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                with tf.variable_scope('Mixed_6d'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

                with tf.variable_scope('Mixed_6e'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                end_points['Mixed_6e'] = net
                # third group inceptions
                with tf.variable_scope('Mixed_7a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                    net = tf.concat([branch_0, branch_1, branch_2], 3)
                with tf.variable_scope('Mixed_7b'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                with tf.variable_scope('Mixed_7c'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = tf.concat([
                            slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                            slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                    with tf.variable_scope('Branch_2'):
                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                        branch_2 = tf.concat([
                            slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                            slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                    with tf.variable_scope('Branch_3'):
                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                    net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                return net, end_points

    def inception_v3(self, inputs,
                     num_classes=1000,
                     is_training=True,
                     dropout_keep_prob=0.8,
                     prediction_fn=slim.softmax,
                     spatial_squeeze=True,
                     reuse=None,
                     scope='InceptionV3'):
        # ===================================
        # 该函数主要包括全局平均池化、softmax和Auxiliary Logits。
        # ===================================
        with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
                net, end_points = self.inception_v3_base(inputs, scope=scope)
                # 接下来是Auxiliary Logits辅助节点分类
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                    aux_logits = end_points['Mixed_6e']
                    with tf.variable_scope('AuxLogits'):
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                     scope='AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1b_1x1')
                        aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                                                 padding='VALID', scope='Conv2d_2a_5x5')
                        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,
                                                 normalizer_fn=None, weights_initializer=trunc_normal(0.01),
                                                 scope='Conv2d_2b_1x1')
                        if spatial_squeeze:
                            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                        end_points['AuxLogits'] = aux_logits
                # 接下来是正常的分类预测逻辑
                with tf.variable_scope('Logits'):
                    net = slim.avg_pool2d(net, [8, 8], scope='AvgPool_1a_8x8')
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                    end_points['PreLogits'] = net
                    logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                         scope='Conv2d_1c_1x1')
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                end_points['Logits'] = logits
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
        return logits, end_points

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
        inceptionv3 = InceptionV3(
            sess,
            batch_size=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            checkpoint_dir=FLAGS.checkpoint_dir)
        inceptionv3.train(FLAGS)