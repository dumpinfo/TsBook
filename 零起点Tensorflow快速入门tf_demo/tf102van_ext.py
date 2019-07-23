#coding=utf-8
'''
Created on 2017.03.03
极宽版·梵高神经网络算法·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip      www.ziwang.com
Top极宽量化开源团队

【说明】
神经网络梵高画风案例（neural-style），与alphaGo阿尔法狗围棋大师，可以说是2016年度，Tensorflow深度学习领域最成功、也是最经典的两个案例。
与庞大复杂的alphaGo阿尔法狗系统不同，梵高案例（neural-style），只需要一台笔记本即可运行，甚至连GPU加速卡都不需要，成为许多TF团队的hello案例。
目前，github已经涌现了大量的衍生版本，甚至淹没了最早的Torch原始版本。
不过，梵高案例（neural-style），对于初学者而言，还是过于复杂，故此，Top极宽量化开源团队，特意推出了一个更加简单的入门版本。


Tensorflow神经网络的neural-style案例
1、案例源自github经典的Tensorflow神经网络neural-style案例
    https://github.com/anishathalye/neural-style
2，考虑初学者门款和教学需要，对源码stylize.py进行了高度精简
3，本案例经过多处优化，可直接运行于纯cpu平台，i7笔记本默认参数3分钟左右
4，参数 ITERATIONS是迭代次数，默认是迭代50，效果很差，500次迭代的效果，基本接近原案例
5，为提高运行速度，对图片进行了缩减，只有原图的1/3，理论上可以提速9倍，原始图像文件保存在pic00目录。
6，输出文件和图像文件都保存在pic目录，输出文件名是'tg_n'+迭代次数
7，模型库文件名，imagenet-vgg-verydeep-19.mat，大约550M

'''
import os
import arrow
import numpy as np
import scipy.misc
from stylize import stylize
import math
from PIL import Image

#-----------------

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000 #1000
VGG_PATH = '/ailib/model/imagenet-vgg-verydeep-19.mat'
POOLING = 'max'


#-----------------
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


#-----------------

#-----------------            
def main(fsr,fsty,ftg,nloop=200):
    #content_image = imread('pic/1-content.jpg')
    #flst=['pic/1-style.jpg']
    content_image = imread(fsr)
    flst=[fsty]
    style_images = [imread(fss) for fss in flst]

    
    width = None
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    #style_blend_weights = options.style_blend_weights
    style_blend_weights = None
    style_blend_weights = [1.0/len(style_images) for _ in style_images]

    #initial = options.initial
    initial =None
    for iteration, image in stylize(
        network=VGG_PATH, #options.network,
        initial=initial,
        initial_noiseblend=1.0,#options.initial_noiseblend,
        content=content_image,
        styles=style_images,
        preserve_colors=True, #options.preserve_colors,
        iterations=nloop,#ITERATIONS,#options.iterations,
        content_weight=CONTENT_WEIGHT,#options.content_weight,
        content_weight_blend=CONTENT_WEIGHT_BLEND,#,options.content_weight_blend,
        style_weight=STYLE_WEIGHT,#options.style_weight,
        style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,#options.style_layer_weight_exp,
        style_blend_weights=style_blend_weights,
        tv_weight=TV_WEIGHT,#options.tv_weight,
        learning_rate=LEARNING_RATE,#options.learning_rate,
        beta1=BETA1,#options.beta1,
        beta2=BETA2,#options.beta2,
        epsilon=EPSILON,#options.epsilon,
        pooling=POOLING,#options.pooling,
        print_iterations=None,#options.print_iterations,
        checkpoint_iterations=None,#options.checkpoint_iterations
    ):
        #output_file = 'pic/tg_n'+str(ITERATIONS)+'.jpg'
        #output_file = 'pic/cx_n'+str(nloop)+'.jpg'
        #ftg = 'pic/cx_n'+str(nloop)+'.jpg'
        combined_rgb = image
        imsave(ftg, combined_rgb)
        print(ftg)
   
#-----------------    

#1
rss,nloop='pic/',200


#2
rlst=['p1','p2','p3','p4']
xlst=['csty1','csty2','csty3','sty1','sty2','sty3','sty4','sty5','sty6','sty7']

#3
for fsr0 in rlst:
    for fsty0 in xlst:
        #3.a
        fsr,fsty=rss+fsr0+'.jpg',rss+fsty0+'.jpg'
        ftg=rss+'x_'+fsr0+'_'+fsty0+'_'+str(nloop)+'.jpg'
        #3.b
        main(fsr,fsty,ftg,nloop)
    

    
'''
tn, 777.98s,nloop, 100
tn, 1510.91 s,nloop, 200
'''