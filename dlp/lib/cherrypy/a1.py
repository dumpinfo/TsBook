import cherrypy

class HelloWorld(object):
    @cherrypy.expose
    def index(self):
        return 'Wky API Server说中文？'

if __name__ == '__main__':
    cherrypy.config.update({
        'server.socket_host': '192.168.1.16',
        'server.socket_port': 8090,
    })
    cherrypy.quickstart(HelloWorld(), '/', {'/': {
    }})

