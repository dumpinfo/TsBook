import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import json
import conf.web_conf as web_conf
import app_global as ag

class CSystem(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir
        
    def exitSystem0(self, params):
        print('exit system')
        ag.task_q_thread.is_stopping = True
        ag.result_q_thread.is_stopping = True
        ag.rdb_pool_cleaner.is_stopping = True
        ag.wdb_pool_cleaner.is_stopping = True
        sys.exit()
        cherrypy.engine.stop()
        
    @staticmethod
    def exitSystem():
        print('exit system')
        sys.exit()

