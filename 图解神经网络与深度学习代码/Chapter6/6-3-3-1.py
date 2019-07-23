# -*- coding: utf-8 -*-
import tensorflow as tf
import os
#"占位符"
x=tf.placeholder(tf.float32,[None,28*28])
labels=tf.placeholder(tf.float32,[None,10])
#"------第一步：解析数据------"
nums=33#"所有训练样本的个数"
#"得到文件夹./data/下的所有tfRecord文件"
files=tf.train.match_filenames_once(os.path.curdir+
                                       "/data/"+
                                       "data*.tfrecord")
#"创建TFRecordReader对象"
num_epochs=1000
reader=tf.TFRecordReader()
records_queue=tf.train.string_input_producer(files,num_epochs=num_epochs)
_,serialized_example=reader.read(records_queue)
#"解析文件中的图像及其对应的标签"
features=tf.parse_single_example(
        serialized_example,
        features={
                'img_raw':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64),
                }
                                 )

#"解码二进制数据"
img_raw=features['img_raw']
img_raw=tf.decode_raw(img_raw,tf.uint8)
img=tf.reshape(img_raw,[28*28])
img=tf.cast(img,tf.float32)
img=img/255.0
#"标签"
label=features['label']
label=tf.cast(label,tf.int64)
label_onehot=tf.one_hot(label,10,dtype=tf.float32)
#"每次从文件中读取3张图片"
BatchSize =3
imgs,labels_onehot=tf.train.shuffle_batch([img,label_onehot],
                        BatchSize,1000+3*BatchSize,1000)
#"------第2部分：构建全连接网络------"
#""输入层、隐含层、输出层的神经元个数""
I,H1,O=784,200,10
#"输入层到隐含层的权重矩阵和偏置"
w1=tf.Variable(tf.random_normal([I,H1],0,1,tf.float32),
                dtype=tf.float32,name='w1')
b1=tf.Variable(tf.random_normal([H1],0,1,tf.float32),
                dtype=tf.float32,name='b1')
#"隐含层的结果，采用 sigmoid 激活函数"
l1=tf.matmul(x,w1)+b1
sigma1=tf.nn.sigmoid(l1)
#"第2层隐含层到输出层的的权重矩阵和偏置"
w2=tf.Variable(tf.random_normal([H1,O],0,1,tf.float32),
                dtype=tf.float32,name='w2')
b2=tf.Variable(tf.random_normal([O],0,1,tf.float32),
                dtype=tf.float32,name='b2')
#"输出层的结果"
logits=tf.matmul(sigma1,w2)+b2
#"------第3部分：构造损失函数------"
loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=labels_onehot,logits=logits))
#"------第4部分：梯度下降------"
opti=tf.train.AdamOptimizer(0.001,0.9,0.999,1e-8).minimize(loss)
#"创建会话"
session=tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=session,coord=coord)
for i in range(num_epochs):
    for n in range(int(nums/BatchSize)):
        imgs_arr,lables_onehot_arr=session.run([imgs,labels_onehot])
        session.run(opti,feed_dict={x:imgs_arr,labels:lables_onehot_arr})
coord.request_stop()
coord.join(threads)
session.close()