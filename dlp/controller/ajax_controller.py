import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import json
import conf.web_conf as web_conf
import app_global as ag

class Ajax_Controller(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir

    @cherrypy.tools.json_out()
    def GET(self, params={}):
        return self.http_method(params)

    @cherrypy.tools.json_out()
    def POST(self, params={}):
        print('params:{0}'.format(params))
        json_obj = json.loads(params['kwargs']['json_str'])
        del params['kwargs']['json_str']
        params['kwargs']['json_obj'] = json_obj
        return self.http_method(params)

    def http_method(self, params):
        filename = params['kwargs'].get('f', 'unknow')
        class_name = params['kwargs'].get('c', 'C')
        method_name = params['kwargs'].get('m', 'M')
        fullname = 'controller.' + filename
        mdl = __import__(fullname, globals(), locals(), [class_name])
        class_t = getattr(mdl, class_name)
        #obj = class_t()
        mtd = getattr(class_t, method_name)
        return mtd(params)
        '''
        obj = sys.modules[fullname]
        func = getattr(obj, cmd)
        return func(params)
        '''


