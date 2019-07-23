import sys
sys.path.append('./lib/cherrypy')
sys.path.append('./lib/tf/slim')
import os
import cherrypy
import app_global as ag
import model.m_mysql as db
import app_web as app_web
import common.wky_queues as wqs

#params = {}
#lcct.startup(params)
#aiw_main.startup(params)
#mcv_main.startup(params)
#fai_main.startup(params)


if __name__ == '__main__':
    print('starting up...')
    db.init_db_pool()
    wqs.init_wky_queues()
    app_web.startup()
    ag.rdb_pool_cleaner.join()
    ag.wdb_pool_cleaner.join()
    ag.task_q_thread.join()
    ag.result_q_thread.join()
