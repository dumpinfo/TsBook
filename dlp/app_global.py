import os
import platform

dataset_dir = os.getcwd() + '/data/'
ann_mf_dir = os.getcwd() + '/repository/'
web_dir = os.getcwd() + '/view/'
upload_dir = os.getcwd().replace('\\', '/') + '/upload/'
resources_dir = os.getcwd().replace('\\', '/') + '/resources/'
if 'Windows' == platform.system():
    jinja2_searchpath = 'D:/'
else:
    jinja2_searchpath = '/'

shared_key = 'wky'

app_db = {}

global task_q
global result_q
global task_q_thread
global result_q_thread
# 图像分类相关
global img_csf_q
global img_csf_q_thread
base_img = os.getcwd() + os.path.sep + 'work/lcct' + os.path.sep + 'cat.jpg'
# 图像识别相关
global img_idf_q
global img_idf_q_thread


