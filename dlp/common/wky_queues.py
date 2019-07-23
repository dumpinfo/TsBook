# -*- coding: utf-8 -*-
import threading
import queue
import app_global as ag
import common.task_queue as tq
import common.result_queue as rq
import common.img_csf_q as iq

def init_wky_queues():
    print('')
    ag.task_q = queue.Queue(maxsize=10)
    ag.result_q = queue.Queue(maxsize=10)
    ag.task_q_thread = tq.Task_Q_Thread()
    ag.task_q_thread.start()
    ag.result_q_thread = rq.Result_Q_Thread()
    ag.result_q_thread.start()
    # 
    ag.img_csf_q = queue.Queue(maxsize=10)
    ag.img_csf_q_thread = iq.Img_Csf_Q_Thread()
    ag.img_csf_q_thread.start()


