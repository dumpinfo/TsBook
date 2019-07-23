import os
import sys
sys.path.append('./lib/cherrypy')
import cherrypy
from cherrypy.lib import auth_digest

g_auth_users = {'yant': '123456'}

app_conf = {'/': {
        'tools.staticdir.root': os.path.abspath('./'),
        'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    },
    '/web/resources': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': 'resources',
    },
    '/web/upload': {
        'tools.staticdir.on': True,
        'tools.staticdir.dir': 'upload',
    },
    '/api': {
        'tools.auth_digest.on': True,
        'tools.auth_digest.realm': '192.168.0.105',
        'tools.auth_digest.get_ha1': auth_digest.get_ha1_dict_plain(g_auth_users),
        'tools.auth_digest.key': 'a565c27146791cfb'
    }
}


