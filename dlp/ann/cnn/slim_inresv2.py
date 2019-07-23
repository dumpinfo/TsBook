import os
import sys
sys.path.append('./lib/tf/models/research/slim')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as FontProperties
from PIL import Image
import tensorflow as tf
from datasets import imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing
import app_global as ag

# *******************************************************************************
# 图像识别类，采用Inception V4和ResNet V2相结合的网络形式，实现Top1精度为80.4，
# Top5精度为95.3
# 闫涛 2016.09.30 初始版本
# *******************************************************************************

class SlimInresV2(object):
    '''
    Inception V2和ResNet混合模型，网络结构定义见in_res_v2.py，原始论文见：
    https://arxiv.org/abs/1602.07261
    实践证明这个网络比Inception V4准确率略有提高，是目前公开的识别精度最
    的模型
    '''
    def __init__(self):
        '''
        我们在这里使用了预训练模型，文件放在models/lcct目录下
        '''
        self.ckpt_dir = './models/'
        self.slim = tf.contrib.slim
        self.batch_size = 3
        self.image_size = inception_resnet_v2.inception_resnet_v2.default_image_size
    
    def startup(self):
        names = self.get_imagenet_label_names()
        with tf.Graph().as_default():
            with self.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                img = tf.placeholder(tf.float32, shape=[299, 299, 3])
                imgs = tf.placeholder(tf.float32, shape=[1, 299, 299, 3])
                logits, _ = inception_resnet_v2.inception_resnet_v2(imgs, num_classes=1001, is_training=False)
                probabilities = tf.nn.softmax(logits)
                init_fn = self.slim.assign_from_checkpoint_fn(
                    os.path.join(self.ckpt_dir, 'inception_resnet_v2_2016_08_30.ckpt'), 
                    self.slim.get_model_variables('InceptionResnetV2')
                )
                input = imgs
                with tf.Session() as sess:
                    init_fn(sess)
                    params = ag.img_csf_q.get(block=True)
                    while params:
                        req_id = params['req_id']
                        testImage_string = tf.gfile.FastGFile(params['img_file'], 'rb').read()
                        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
                        processed_image = inception_preprocessing.preprocess_image(testImage, self.image_size, self.image_size, is_training=False)
                        processed_images = tf.expand_dims(processed_image, 0)
                        np_image, rst_p = sess.run([imgs, probabilities], feed_dict={imgs: processed_images.eval()})
                        pros = rst_p[0, 0:]
                        sorted_inds = [i[0] for i in sorted(enumerate(-pros), key=lambda x: x[1])]
                        index = sorted_inds[0]
                        rst = {}
                        rst['req_id'] = req_id
                        rst['img_rst'] = names[index]
                        rst['probability'] = pros[index]
                        ag.app_db[req_id] = rst
                        print('******运行Tensorflow成功:{0}({1})'.format(rst['img_rst'], rst['probability']))
                        params = ag.img_csf_q.get(block=True)
                        
                        
                        
    
    def predict(self, img_file):
        '''
        对给定的图片进行预测，给出Top5类别的名称及可能的概率
        img_file：全路径文件名
        '''
        print('image file:{0}'.format(img_file))
        names = self.get_imagenet_label_names()
        with tf.Graph().as_default():
            with self.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                testImage_string = tf.gfile.FastGFile(img_file, 'rb').read()
                testImage = tf.image.decode_jpeg(testImage_string, channels=3)
                processed_image = inception_preprocessing.preprocess_image(testImage, self.image_size, self.image_size, is_training=False)
                processed_images = tf.expand_dims(processed_image, 0)
                logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
                probabilities = tf.nn.softmax(logits)
                init_fn = self.slim.assign_from_checkpoint_fn(
                    os.path.join(self.ckpt_dir, 'inception_resnet_v2_2016_08_30.ckpt'), 
                    self.slim.get_model_variables('InceptionResnetV2')
                )
                with tf.Session() as sess:
                    init_fn(sess)
                    np_image, probabilities = sess.run([processed_images, probabilities])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]
                    #names = imagenet.create_readable_names_for_imagenet_labels()
                    for i in range(5):
                        index = sorted_inds[i]
                        print((probabilities[index], names[index]))
                    print('show:{0}'.format(img_file))
                    img = np.array(Image.open(img_file))
                    plt.rcParams['font.sans-serif'] = ['SimHei'] # 在图片上显示中文，如果直接显示形式为：u'内容'
                    plt.rcParams['axes.unicode_minus'] = False # 显示-号
                    plt.figure()
                    plt.imshow(img)
                    myfont = FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc')
                    plt.suptitle(names[sorted_inds[0]], fontsize=14, fontweight='bold',fontproperties=myfont)
                    plt.axis('off')
                    plt.show()
                
                
    def get_imagenet_label_names(self):
        '''
        imagenet类别的名称，因为国内网络问题，我们将其内容下载到本地文件，有些条目
        已经翻译成中文
        '''
        import json
        names = {}
        for line in open('./ann/imagenet_label_names.txt', 'r', encoding='utf-8'):
            if len(line) > 3:
                items = line.split(': \'')
                key = int(items[0])
                vs = items[1].split('\',')
                val = vs[0]
                names[key] = val
        return names
