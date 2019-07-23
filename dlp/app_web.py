import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import json
import conf.web_conf as web_conf
from controller.page_controller import Page_Controller
from controller.ajax_controller import Ajax_Controller
from controller.web_controller import Web_Controller
from controller.upload_controller import File_Uploader

def startup():
    cherrypy.config.update({
        'server.socket_host': '0.0.0.0',
        'server.socket_port': 8090,
    })
    page_controller = Page_Controller()
    ajax_controller = Ajax_Controller()
    web_controller = Web_Controller()
    upload_controller = File_Uploader()
    cherrypy.tree.mount(page_controller, '/web/pages', web_conf.app_conf)
    cherrypy.tree.mount(web_controller, '/', web_conf.app_conf)
    cherrypy.tree.mount(ajax_controller, '/wky/dl/mlp', web_conf.app_conf)
    cherrypy.tree.mount(ajax_controller, '/ajax', web_conf.app_conf)
    cherrypy.tree.mount(upload_controller, '/upload', {'/upload': {}})
    cherrypy.engine.start()
    #cherrypy.engine.block()


