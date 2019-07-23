import sys
sys.path.append('./lib/cherrypy')
import cherrypy
import app_global as ag
import conf.pages_conf as pages_conf

class Page_Controller(object):
    exposed = True
    def __init__(self):
        self.web_dir = ag.web_dir
        self.pages_conf = pages_conf.pages_conf

    def GET(self, params={}):
        print('??????????????????????????????????????? web page ?????????????')
        cmd = params['kwargs']['cmd']
        return self._show_page(params)
        #func = getattr(self, cmd)
        #return func(params)

    def read_html(self, file):
        fo = open(self.web_dir + file, 'r', encoding='utf-8')
        try:
            html = fo.read()
        finally:
            fo.close()
        return html

    '''
    def show_rest01(self, params):
        cmd = params['kwargs']['cmd']
        page_conf = self.pages_conf[cmd]
        header = ''
        if '' != page_conf['header']:
            header = self.read_html(page_conf['header'])
        left = ''
        if '' != page_conf['left']:
            left = self.read_html(page_conf['left'])
        body = ''
        if '' != page_conf['body']:
            body = self.read_html(page_conf['body'])
        right = ''
        if '' != page_conf['right']:
            right = self.read_html(page_conf['right'])
        footer = ''
        if '' != page_conf['footer']:
            footer = self.read_html(page_conf['footer'])
        return header + left + body + right + footer
    '''

    def show_img_csf(self, params):
        '''
        显示图像识别页面
        '''
        return self.read_html('tabbar.html')

    def _show_page(self, params):
        cmd = params['kwargs']['cmd']
        page_conf = self.pages_conf[cmd]
        header = ''
        if '' != page_conf['header']:
            header = self.read_html(page_conf['header'])
        left = ''
        if '' != page_conf['left']:
            left = self.read_html(page_conf['left'])
        body = ''
        if '' != page_conf['body']:
            body = self.read_html(page_conf['body'])
        right = ''
        if '' != page_conf['right']:
            right = self.read_html(page_conf['right'])
        footer = ''
        if '' != page_conf['footer']:
            footer = self.read_html(page_conf['footer'])
        return header + left + body + right + footer


