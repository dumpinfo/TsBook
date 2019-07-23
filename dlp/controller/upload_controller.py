import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import cgi
import tempfile
import os
import shutil
import time
from PIL import Image as image
import pickle
import numpy as np
import app_global as ag
import common.wky_auth as wky_auth
import common.img_util as img_util

class Field_Storage(cgi.FieldStorage):
    def make_file(self, binary=None):
        return tempfile.NamedTemporaryFile()

def no_body_process():
    cherrypy.request.process_request_body = False

cherrypy.tools.noBodyProcess = cherrypy.Tool('before_request_body', \
                                      no_body_process)
cherrypy.server.socket_timeout = 60
cherrypy.server.max_request_body_size = 0

class File_Uploader(object):
    @cherrypy.expose
    def index(self, params={}):
        return 'File_Uploader'

    @cherrypy.expose
    @cherrypy.tools.noBodyProcess()
    @cherrypy.tools.json_out()
    def upload_mlp1(self, params={}, data_file=None):
        print('########### upload image classification file:%s' % params)
        #if not wky_auth.validate_req(params['kwargs']):
        #    return {'status': 'Error'}
        dest_file, uploaded_file, disp_file, raw_disp_file, req_id, user_id = self.upload_base('files[]', is_img=1, img_size=(200, 200))
        resp_params = {}
        resp_params['req_id'] = req_id
        resp_params['user_id'] = user_id
        resp_params['img_file'] = dest_file
        print('将图像加入到图像分类任务队列...')
        ag.img_csf_q.put(resp_params)
        resp = {'status': 'Ok', 'req_id': req_id, 'url': '/web/upload/' + raw_disp_file}
        return resp
        
    def upload_img_idf_file(self, params={}, data_file=None):
        print('##### upload image identification file:{0}'.format(params))
        dest_file, uploaded_file, disp_file, raw_disp_file, req_id, user_id = self.upload_base('files[]', is_img=1, img_size=(200, 200))
        resp_params = {}
        resp_params['req_id'] = req_id
        resp_params['user_id'] = user_id
        resp_params['img_file'] = dest_file
        print('将图像加入到图像识别任务队列...')
        ag.img_csf_q.put(resp_params)
        resp = {'status': 'Ok', 'req_id': req_id, 'url': '/web/upload/' + raw_disp_file, 'org_file': dest_file}
        return resp
        
    def upload_base(self, file_elem_name, is_img=1, img_size=(200, 200)):
        cherrypy.response.timeout = 3600
        req_headers = {}
        for hdr in cherrypy.request.headers.items():
            req_headers[hdr[0].lower()] = hdr[1]
        form_fields = Field_Storage(fp=cherrypy.request.rfile, \
                             headers = req_headers, environ = {\
                                          'REQUEST_METHOD': 'POST'},\
                                          keep_blank_values=True)
        # print(form_fields.__dict__)
        data_file = form_fields[file_elem_name] # 'files[]']
        req_id = form_fields['req_id'].value
        user_id = form_fields['user_id'].value
        src_file = data_file.file.name
        uploaded_file = 'f_' + \
                    str(time.time()).replace('.', '_') + \
                    os.path.splitext(data_file.filename)[1]
        raw_disp_file = 'f_' + \
                    str(time.time()).replace('.', '_') + \
                    '_disp' + \
                    os.path.splitext(data_file.filename)[1]
        dest_file = ag.upload_dir + uploaded_file
        disp_file = ag.upload_dir + raw_disp_file
        uf = open(dest_file, 'wb')
        uf.write(data_file.file.read())
        uf.close()
        if 1 == is_img:
            img_util.resize_img_file(dest_file, disp_file, img_size[0], img_size[1], save_quality=35)
        return (dest_file, uploaded_file, disp_file, raw_disp_file, req_id, user_id)
        
    
