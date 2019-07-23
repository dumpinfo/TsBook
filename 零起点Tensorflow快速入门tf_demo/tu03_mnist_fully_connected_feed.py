#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

简单的MNIST手写字体识别案例
使用 FFNNs 前馈神经网络模型

@from:
https://www.tensorflow.org//examples /tutorials/mnist/

'''

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist



def placeholder_inputs(batch_size):
  """
   生成placeholder占位符变量，用于表示输入张量。
   这些placeholder占位符变量，用于算法模型的输入数据指针，会在run运行函数当中，替换为实际的数据
   输入参数：
   batch_size：batch_size批量的大小，会被复制到两个placeholder占位符变量。
   返回值：
   images_placeholder：图像占位符。
   labels_placeholder：标签的占位符。  
  """
  
  #注意placeholders占位符变量的shape形状匹配数据，是图像和和标签的tensors张量数据
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """
    根据给定训练步骤，填充feed_dict数据包。
    feed_dict数据包格式：
    feed_dict = {
            <placeholder占位符>：要传递给位符值的tensor张量数值>，
            …更多的<placeholder占位符变量>
        }
    输入参数
    data_set：图像和标签数据集，源自input_data.read_data_sets()
    images_pl：图像占位符，源自placeholder_inputs()。
    labels_pl：标签的占位符，源自placeholder_inputs()。
    返回值：
    feed_dict：feed数据集，源自placeholders占位符对应的数据。
    
  """
  # 创建feed_dict数据，用于填充下一组 `batch size`批量的placeholders占位符数据
  images_feed, labels_feed = data_set.next_batch(100,False)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """
      对一轮完整的训练，进行一次评估。
      输入参数：
      sess：会话变量，用于正在进行的模型训练。
      eval_correct：返回正确的预测数张量Tensor数目。
      images_placeholder：图像占位符。
      labels_placeholder：标签的占位符。
      data_set：图像和标签数据集，源自 input_data。read_data_sets()。

  """
  # 评估一次迭代
  true_count = 0  # 计数器，正确的预测数据。
  steps_per_epoch = data_set.num_examples // 100
  num_examples = steps_per_epoch * 100
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


#---------main
#1
print('\n#1,set.dat')
#读取MNIST数据集，包括training训练, validation验证, test测试数据集.
rlog='/ailib/log_tmp/'
#data_sets = input_data.read_data_sets('data/mnist', False)
data_sets = input_data.read_data_sets('data/mnist')

#2
print('\n#2,使用default Graph默认图结构，构造模型')
with tf.Graph().as_default():
  #3
  print('\n#3,设置输入placeholder占位符变量参数，用于图像和对应的标签')
  images_placeholder, labels_placeholder = placeholder_inputs(100)

  
  #4
  print('\n#4,构建Graph结构图，使用mnist模块inference模型计算预测数据')
  logits = mnist.inference(images_placeholder,128,32)

  
  #5
  print('\n#5,增加一个Ops操作，用于定义loss损失函数')
  loss = mnist.loss(logits, labels_placeholder)

  #6
  print('\n#6,增加一个Ops操作，用于计算训练梯度')
  train_op = mnist.training(loss, 0.01)

  
  #7
  print('\n#7,增加一个Ops操作，用于比较标签占位符和预测结果')
  eval_correct = mnist.evaluation(logits, labels_placeholder)

  
  #8
  print('\n#8,合并输出summary所有日志参数')
  summary = tf.summary.merge_all()

  #9
  print('\n#9,定义初始化操作变量参数')
  init = tf.global_variables_initializer()

  
  #10
  print('\n#10,创建checkpoints检查点变量，用于保存模型数据')
  saver = tf.train.Saver()

  
  #11
  print('\n#11,创建Session会话参数')
  sess = tf.Session()

  
  #12
  print('\n#12,输出shummary日志graph运算结构图数据')
  summary_writer = tf.summary.FileWriter(rlog, sess.graph)

  #13
  print('\n#13,完成全部预备工作')
  print('执行变量初始化操作op')
  sess.run(init)

  
  #14
  print('\n#14,开始迭代训练')
  nsteps=2000 #0
  for step in range(nsteps):
    start_time = time.time()

    
    #15 
    # 从训练数据集，提取数据，用于本轮迭代训练，数据为dict字典格式
    feed_dict = fill_feed_dict(data_sets.train,images_placeholder,labels_placeholder)

    
    
    #16
    #根据模型，输入数据，进行一轮训练学习
    #返回值是，train_op训练函数和loss损失函数的计算数值
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)

    duration = time.time() - start_time

    
    #17
    # 输出中间数据和保存summary日志数据
    if step % 100 == 0:
      # 输出中间数据，序号，loss损失数据，本轮运算时间
      print('\nStep %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      
      # 更新各种变量的summary日志数据，并保存
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

    
    #18
    #定期保存checkpoint检查点数据，并对模型进行评估。
    if (step + 1) % 100 == 0 or (step + 1) == nsteps:
      #19  
      checkpoint_file = rlog+ 'model.ckpt'
      saver.save(sess, checkpoint_file, global_step=step)
      
      #20
      print('\n step#',step)
      print('Train训练数据集:')
      do_eval(sess,
              eval_correct,
              images_placeholder,
              labels_placeholder,
              data_sets.train)
      #21
      print('validation验证数据集:')
      do_eval(sess,
              eval_correct,
              images_placeholder,
              labels_placeholder,
              data_sets.validation)
      
      #22
      print('Test测试数据集:')
      do_eval(sess,
              eval_correct,
              images_placeholder,
              labels_placeholder,
              data_sets.test)

