# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
#ckpt=tf.train.get_checkpoint_state('./')
#"获取在当前文件夹下(./)的ckpt文件，具体根据ckpt保存的位置设置"
ckpt=tf.train.latest_checkpoint('./')
#"打印获取的ckpt文件"
print('获取的ckpt文件:'+ckpt)
#"创建NewCheckpointReader类,读取ckpt文件中的变量名称及其对应的值"
reader=pywrap_tensorflow.NewCheckpointReader(ckpt)
var_to_shape_map=reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print('tensor_name:',key)
    print(reader.get_tensor(key))