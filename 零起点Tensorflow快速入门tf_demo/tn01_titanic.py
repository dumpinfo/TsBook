#coding=utf-8
'''
Created on 2017.03.03
极宽版·深度学习·案例
摘自·极宽深度学习·系列培训课件
@ www.TopQuant.vip    www.ziwang.com
Top极宽量化开源团队

titanic数据集
使用TFlearn简化接口 

@from:
tflearn.org
'''



import numpy as np
import tflearn,os
import tensorflow as tf
from tflearn.datasets import titanic
from tflearn.data_utils import load_csv
#

#--函数定义 数据预处理
def preprocess(passengers, columns_to_delete):
    # 对数据采用id排序
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [passenger.pop(column_to_delete) for passenger in passengers]
    for i in range(len(passengers)):
        # 转换sex字典数据为float浮点数
        passengers[i][1] = 1. if passengers[i][1] == 'female' else 0.
    return np.array(passengers, dtype=np.float32)
#---------------

#1
print('\n#1,set.dat')
rlog='/ailib/log_tmp'
if os.path.exists(rlog):tf.gfile.DeleteRecursively(rlog)



#2.1
print('\n#2.1,get.titanic.dat')
fdat='data/titanic_dataset.csv'
titanic.download_dataset(fdat)

#2.2
print('\n#2.2,加载CSV文件，指示第一列表示标签')
data, labels = load_csv(fdat, target_column=0,categorical_labels=True, n_classes=2)

#2.3
print('\n#2.3,数据筛选，呼市字段 name 和 ticket 之间的六组字段数据')
to_ignore=[1, 6]

#2.4
print('\n#2.4,调用preprocess数据预处理函数，整理数据')
data = preprocess(data, to_ignore)


#3.1
print('\n#3.1,构建神经网络模型')
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)


#3.1
print('\n#3.1,构建DNN网络模型')
model = tflearn.DNN(net,tensorboard_dir=rlog)


#4
print('\n#4,开始训练模型')
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

#5
print('\n#5,计算分析结果')

#5.1
print('\n#5.1,分析数据')
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

#5.2
print('\n#5.2,数据预处理')
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)

#5.3
print('\n#5.3,计算结果：生存机率')
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
