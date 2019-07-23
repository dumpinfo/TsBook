import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import json
import conf.web_conf as web_conf
import app_global as ag

class CImgCsf(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir
        
    @staticmethod
    def get_img_csf_rst(params):
        req_id = params['kwargs']['req_id']
        has_rst = 0
        rst = '未知'
        if req_id in ag.app_db:
            params = ag.app_db[req_id]
            rst = params['img_rst']
            has_rst = 1
            del ag.app_db[req_id]
        print('req_id={0}'.format(req_id))
        resp = {'status': 'Ok', 'req_id': req_id, 'img_rst': rst}
        resp['has_rst'] = has_rst
        return resp

