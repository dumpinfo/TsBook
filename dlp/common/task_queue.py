import threading
import queue
import app_global as ag
#import controller.c_mlp as c_mlp

class Task_Q_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_stopping = False

    def run(self):
        params = ag.task_q.get(block=True)
        while params:
            #result = c_mlp.classify_img(params)
            if self.is_stopping:
                return
            result = {}
            result['user_id'] = params['user_id']
            result['req_id'] = params['req_id']
            result['order_id'] = result['user_id'] + '_' + result['req_id']
            result['img_rst'] = '最成功的虎斑猫'
            ag.result_q.put(result)
            params = ag.task_q.get(block=True)


