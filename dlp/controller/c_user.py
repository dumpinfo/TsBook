import json
from model.mf_user import MFUser as MFUser

class CUser(object):
    @staticmethod
    def login_user(req_args):
        ''' 获取题目的HTML内容 '''
        params = req_args['kwargs']['json_obj']
        login_name = params['login_name']
        login_pwd = params['login_pwd']
        user_id, user_name, role_id = MFUser.login_user(login_name, login_pwd)
        resp = {}
        if user_id > 0:
            resp['status'] = 'Ok'
        else:
            resp['status'] = 'Error'
        resp['user_id'] = user_id
        resp['user_name'] = user_name
        resp['role_id'] = role_id
        return resp
        