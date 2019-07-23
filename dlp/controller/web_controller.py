import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import app_global as ag

class Web_Controller(object):
    def __init__(self):
        self.web_dir = ag.web_dir

    @cherrypy.expose
    def index(self):
        print('web_controller is running...')
        return 'Wky Deep Learning Cloud API'


