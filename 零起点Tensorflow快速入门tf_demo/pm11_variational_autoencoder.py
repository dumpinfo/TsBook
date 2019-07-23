#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

pkmital案例合集

@from:
pkmital案例合集网址：
https://github.com/pkmital/tensorflow_tutorials

'''
import tensorflow as tf
import numpy as np
from libs.utils import weight_variable, bias_variable, montage_batch
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
#----------------------------

#10 定义 VAE变分自编码器函数
def VAE(input_shape=[None, 784],
#def VAE(input_shape=[10000, 784],
        n_components_encoder=2048,
        n_components_decoder=2048,
        n_hidden=2,
        debug=False):
    
    #10.1输入参数，占位符
    if debug:
        input_shape = [50, 784]
        x = tf.Variable(np.zeros((input_shape), dtype=np.float32))
    else:
        x = tf.placeholder(tf.float32, input_shape)

    activation = tf.nn.softplus

    dims = x.get_shape().as_list()
    n_features = dims[1]

    W_enc1 = weight_variable([n_features, n_components_encoder])
    b_enc1 = bias_variable([n_components_encoder])
    h_enc1 = activation(tf.matmul(x, W_enc1) + b_enc1)

    W_enc2 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc2 = bias_variable([n_components_encoder])
    h_enc2 = activation(tf.matmul(h_enc1, W_enc2) + b_enc2)

    W_enc3 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc3 = bias_variable([n_components_encoder])
    h_enc3 = activation(tf.matmul(h_enc2, W_enc3) + b_enc3)

    W_mu = weight_variable([n_components_encoder, n_hidden])
    b_mu = bias_variable([n_hidden])

    W_log_sigma = weight_variable([n_components_encoder, n_hidden])
    b_log_sigma = bias_variable([n_hidden])

    z_mu = tf.matmul(h_enc3, W_mu) + b_mu
    z_log_sigma = 0.5 * (tf.matmul(h_enc3, W_log_sigma) + b_log_sigma)

    #10.2 噪声分布样品，范围：P（EPS）~ N（0，1）
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_hidden])
    else:
        epsilon = tf.random_normal(
            tf.stack([tf.shape(x)[0], n_hidden]))

    #10.3 尾部样本
    z = z_mu + tf.exp(z_log_sigma) * epsilon

    W_dec1 = weight_variable([n_hidden, n_components_decoder])
    b_dec1 = bias_variable([n_components_decoder])
    h_dec1 = activation(tf.matmul(z, W_dec1) + b_dec1)

    W_dec2 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec2 = bias_variable([n_components_decoder])
    h_dec2 = activation(tf.matmul(h_dec1, W_dec2) + b_dec2)

    W_dec3 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec3 = bias_variable([n_components_decoder])
    h_dec3 = activation(tf.matmul(h_dec2, W_dec3) + b_dec3)

    W_mu_dec = weight_variable([n_components_decoder, n_features])
    b_mu_dec = bias_variable([n_features])
    y = tf.nn.sigmoid(tf.matmul(h_dec3, W_mu_dec) + b_mu_dec)

    #10.4 概率函数计算p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    #10.5 概率函数计算，d_kl(q(z|x)||p(z))
    # 附加数据 B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
        
    # 10.6 计算loss损失函数     
    loss = tf.reduce_mean(log_px_given_z + kl_div)
    
    # 10.6 返回数据
    return {'cost': loss, 'x': x, 'z': z, 'y': y}


#20 mnist测试函数
def test_mnist():
		#使用MNIST数据集，测试VAE变分自编码器
    """Summary

    Returns返回值
    -------
    name : TYPE类型
    """
    
    
    #20.1，读取 MNIST 数据
    mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
    
    #20.2 设置VAE变分自编码器ae
    ae = VAE()

    #20.3 设置优化函数
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    #20.4
    print('\n#20.4,设置Session变量，初始化所有graph图计算的所有变量')
    print('使用summary日志函数，保存graph图计算结构图')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rlog='/ailib/log_tmp'
    xsum= tf.summary.FileWriter(rlog, sess.graph) 
    
    #20.5，
    print('\n#20.5,开始训练，迭代次数n_epochs=5，每次训练批量batch_size = 1000')
    t_i = 0
    #batch_size = 100
    #n_epochs = 50
    batch_size = 1000
    n_epochs = 5
    #================
    n_examples = 20
    test_xs, _ = mnist.test.next_batch(n_examples)
    xs, ys = mnist.test.images, mnist.test.labels
    fig_manifold, ax_manifold = plt.subplots(1, 1)
    fig_reconstruction, axs_reconstruction = plt.subplots(2, n_examples, figsize=(10, 2))
    fig_image_manifold, ax_image_manifold = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        #20.5a 按batch_size批量训练模型
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train_cost += sess.run([ae['cost'], optimizer],
                                   feed_dict={ae['x']: batch_xs})[0]
            if batch_i % 2 == 0:
                #20.5a1 从隐藏层，根据模型，绘制再造的图像
                imgs = []
                for img_i in np.linspace(-3, 3, n_examples):
                    for img_j in np.linspace(-3, 3, n_examples):
                        z = np.array([[img_i, img_j]], dtype=np.float32)
                        recon = sess.run(ae['y'], feed_dict={ae['z']: z})
                        imgs.append(np.reshape(recon, (1, 28, 28, 1)))
                imgs_cat = np.concatenate(imgs)
                ax_manifold.imshow(montage_batch(imgs_cat))
                fig_manifold.savefig('tmp/manifold_%08d.png' % t_i)

                #20.5a2 根据模型，绘制再造的图像
                recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
                print(recon.shape)
                for example_i in range(n_examples):
                    axs_reconstruction[0][example_i].imshow(
                        np.reshape(test_xs[example_i, :], (28, 28)),
                        cmap='gray')
                    axs_reconstruction[1][example_i].imshow(
                        np.reshape(
                            np.reshape(recon[example_i, ...], (784,)),
                            (28, 28)),
                        cmap='gray')
                    axs_reconstruction[0][example_i].axis('off')
                    axs_reconstruction[1][example_i].axis('off')
                fig_reconstruction.savefig('tmp/reconstruction_%08d.png' % t_i)

                #20.5a3 绘制隐藏层的流行manifold
                zs = sess.run(ae['z'], feed_dict={ae['x']: xs})
                ax_image_manifold.clear()
                ax_image_manifold.scatter(zs[:, 0], zs[:, 1],
                    c=np.argmax(ys, 1), alpha=0.2)
                ax_image_manifold.set_xlim([-6, 6])
                ax_image_manifold.set_ylim([-6, 6])
                ax_image_manifold.axis('off')
                fig_image_manifold.savefig('tmp/image_manifold_%08d.png' % t_i)

                t_i += 1

				#20.5b 计算每次迭代的相关参数    	
        print('Train cost:', train_cost /
              (mnist.train.num_examples // batch_size))

        valid_cost = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, _ = mnist.validation.next_batch(batch_size)
            valid_cost += sess.run([ae['cost']],
                                   feed_dict={ae['x']: batch_xs})[0]
        print('Validation cost:', valid_cost /
              (mnist.validation.num_examples // batch_size))



if __name__ == '__main__':
    test_mnist()
