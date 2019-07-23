import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


class SVM():
    def __init__(self):
        self.x=tf.placeholder('float',shape=[None,2],name='x_batch')
        self.y=tf.placeholder('float',shape=[None,1],name='y_batch')
        self.sess=tf.Session()

    def creat_dataset(self,size, n_dim=2, center=0, dis=2, scale=1, one_hot=False):
        center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
        center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
        cluster1 = (np.random.randn(size, n_dim) + center1) * scale
        cluster2 = (np.random.randn(size, n_dim) + center2) * scale
        x_data = np.vstack((cluster1, cluster2)).astype(np.float32)
        y_data = np.array([1] * size + [-1] * size)
        indices = np.random.permutation(size * 2)
        x_data, y_data = x_data[indices], y_data[indices]
        y_data=np.reshape(y_data,(y_data.shape[0],1))
        if not one_hot:
            return x_data, y_data
        y_data = np.array([[0, 1] if label == 1 else [1, 0] for label in y_data], dtype=np.int8)
        return x_data, y_data

    @staticmethod
    def get_base(self,_nx, _ny):
        _xf = np.linspace(self.x_min, self.x_max, _nx)
        _yf = np.linspace(self.y_min, self.y_max, _ny)
        n_xf, n_yf = np.meshgrid(_xf, _yf)
        return _xf, _yf,np.c_[n_xf.ravel(), n_yf.ravel()]

    def train(self,step,x_data,y_data):

        w = tf.Variable(np.ones([2,1]), dtype=tf.float32, name="w_v")
        b = tf.Variable(0., dtype=tf.float32, name="b_v")


        self.y_pred =tf.matmul(self.x,w)+b

        cost = tf.nn.l2_loss(w)+tf.reduce_sum(tf.maximum(1-self.y*self.y_pred,0))
        train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

        self.y_predict =tf.sign( tf.matmul(self.x,w)+b )
        self.sess.run(tf.global_variables_initializer())
        for i in range(step):
            index=np.random.permutation(y_data.shape[0])
            x_data1, y_data1 = x_data[index], y_data[index]
            self.sess.run(train_step,feed_dict={self.x:x_data1[0:50],self.y:y_data1[0:50]})
            self.y_predict_value,self.w_value,self.b_value,cost_value=self.sess.run([self.y_predict,w,b,cost],feed_dict={self.x:x_data,self.y:y_data})
            if i%1000==0:print('cost=%f'%cost_value)
    def predict(self,y_data):

        correct = tf.equal(self.y_predict_value, y_data)

        precision=tf.reduce_mean(tf.cast(correct, tf.float32))

        precision_value=self.sess.run(precision)
        return precision_value, self.y_predict_value

    def drawresult(self,x_data):

        self.x_min, self.y_min = np.minimum.reduce(x_data,axis=0) -2
        self.x_max, self.y_max = np.maximum.reduce(x_data,axis=0) +2

        xf, yf , matrix_= self.get_base(self,200, 200)


        print(self.w_value,self.b_value)
        z=np.sign(np.matmul(matrix_,self.w_value)+self.b_value).reshape((200,200))
        plt.pcolormesh(xf, yf, z, cmap=plt.cm.Paired)

        for i in range(x_data.shape[0]):

            if self.y_predict_value[i,0]==1.0:
                plt.scatter(x_data[i,0],x_data[i,1],color='r')
            else:
                plt.scatter(x_data[i,0],x_data[i,1],color='g')

        plt.axis([self.x_min,self.x_max,self.y_min ,self.y_max])
#        plt.contour(xf, yf, z)
        plt.show()

svm=SVM()
x_data,y_data=svm.creat_dataset(size=200, n_dim=2, center=0, dis=4,  one_hot=False)

svm.train(5000,x_data,y_data)
precision_value,y_predict_value=svm.predict(y_data)
svm.drawresult(x_data)