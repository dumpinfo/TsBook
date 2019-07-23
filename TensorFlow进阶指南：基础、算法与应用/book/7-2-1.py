import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(para):
    initial = tf.truncated_normal(para, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(para):
    initial = tf.constant(0.1, shape=para)
    return tf.Variable(initial)

def nn_layer(in_data, in_dim, out_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([in_dim, out_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([out_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(in_data, weights) + biases
            tf.summary.histogram('Wx_plus_b', Wx_plus_b)
        activations = act(Wx_plus_b, name='activation')
        tf.summary.histogram('activations', activations)
        return activations
# 定义原始数据输入
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='x_input')

# 原始数据reshape
with tf.name_scope('input_reshape'):
    x_shaped = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_shaped, 10)

# 创建第一层layer
layer1 = nn_layer(x, 784, 500, 'layer1')
with tf.name_scope('dropout'):  # layer1 后面follow一层dropout
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(layer1, keep_prob)

# 创建第二层layer
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 计算交叉熵cross_entropy
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)  # diff
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy_op)

# 汇总所有summary
merge_op = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/mnist/train', sess.graph)  # 保存log文件路径
test_writer = tf.summary.FileWriter('./logs/mnist/test', sess.graph)  # 保存log文件路径

tf.global_variables_initializer().run()  # 初始化所有变量

# 使用tf.train.Saver()创建模型的保存器
saver = tf.train.Saver()
for i in range(1000):
    if i % 10 == 0:  # for test，每10个step保存一次
        # 读入数据
        xs, ys = mnist.test.images, mnist.test.labels
        # 执行merge_op(数据汇总) 和 accuracy_op(准确率测试)
        summary, acc = sess.run([merge_op, accuracy_op], feed_dict={x: xs, y_: ys, keep_prob: 1.0})
        print('Accuracy at step %s: %s' % (i, acc))  # 打印输出结果

    else:  # for train
        if i == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata_op = tf.RunMetadata()
            xs, ys = mnist.train.next_batch(100)
            summary, _ = sess.run([merge_op, train_step], feed_dict={x: xs, y_: ys, keep_prob: 0.6}, options=run_options, run_metadata=run_metadata_op)
            train_writer.add_run_metadata(run_metadata_op, 'step % 03d', i)
            train_writer.add_summary(summary, i)
            saver.save(sess, './logs/model.ckpt', i)
            print('Adding run metadata for', i)
        else:
            xs, ys = mnist.train.next_batch(100)
            # 执行merged_op和train_step，记录summary
            summary, _ = sess.run([merge_op, train_step], feed_dict={x: xs, y_: ys, keep_prob: 0.6})
            train_writer.add_summary(summary, i)
        # 训练完毕，关闭writer
train_writer.close
test_writer.close