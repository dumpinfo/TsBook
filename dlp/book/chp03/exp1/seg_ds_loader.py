import numpy as np
from matplotlib import pyplot as plt
import csv
from ds_loader import Ds_Loader

class Seg_Ds_Loader(Ds_Loader):
    def load_dataset(self, filename, num_labels):
        ''' 从数据集文件中读出数据集设计矩阵X和标签y
        '''
        X = []
        y_ = []
        # 从训练样本集中读出训练样本和标签
        ds_reader = csv.reader(open(filename, encoding='utf-8'))
        for row in ds_reader:
            y_.append(int(row[0]))
            X.append([float(x) for x in row[1:]])
        # 将特征变为矩阵
        X = np.matrix(X).astype(np.float32)
        # 将标签变为数组，并转化为[1,0]或[0,1]形式，分别代表第一类和第二类
        y_np = np.array(y_).astype(dtype=np.uint8)
        y = (np.arange(num_labels) == y_np[:, None]).astype(np.float32)
        # 返回数据集，以设计矩阵和结果标签形式，形状为：m*n，m*class_num
        return X, y
        
    def prepare_datesets(self, dataset_file, test_file):
        ''' 将train_file分train_ds和validation_ds两个文件，分别占80%和20%,且为随机分配，test_file为测试数据集
        '''
        train_file = 'datasets/train.csv'
        validation_file = 'datasets/validation.csv'
        train_reader = csv.reader(open(dataset_file, encoding='utf-8'))
        train_writer = csv.writer(open(train_file, 'w', newline=''))
        validation_writer = csv.writer(open(validation_file, 'w', newline=''))
        for row in train_reader:
            rand_num = np.random.uniform(0.0, 1.0, 1)
            item = [x for x in row]
            if rand_num[0] > 0.12:
                train_writer.writerow(row)
            else:
                validation_writer.writerow(row)
        return train_file, validation_file, test_file
        
    def draw_train_dataset(self, train_file):        
        x01 = []
        x02 = []
        x11 = []
        x12 = []
        ds_reader = csv.reader(open(train_file, encoding='utf-8'))
        for row in ds_reader:
            if int(row[0]) > 0:
                x11.append(float(row[1]))
                x12.append(float(row[2]))
            else:
                x01.append(float(row[1]))
                x02.append(float(row[2]))
        #draw_scatter(x01, x02, 'r')
        plt.scatter(x01, x02, s=20, color='r')
        plt.scatter(x11, x12, s=20, color='b')
        plt.show()