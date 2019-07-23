import common.wky_auth as wky_auth
import app_global as ag

def async_classify_img(params):
    kwargs = params['kwargs']
    if not wky_auth.validate_req(kwargs):
        return {'status': 'Error'}
    print('异步任务控制器接收到新的异步任务')
    ag.task_q.put(params)
    resp = {'status': 'Ok'}
    return resp


