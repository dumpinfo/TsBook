import threading
import queue
import app_global as ag

class Result_Q_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_stopping = False
        print('初始化结果队列线程')

    def run(self):
        print('启动结果队列处理线程')
        result = ag.result_q.get(block=True)
        while result:
            print('发送结果，保存结果到ag.app_db：%s' % result)
            if self.is_stopping:
                return
            ag.app_db[result['req_id']] = result
            result = ag.result_q.get(block=True)


