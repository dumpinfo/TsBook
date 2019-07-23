#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.append('./lib/cherrypy')
sys.path.append('./lib/jinja')
sys.path.append('./lib/tf/slim')
import os
import cherrypy
import app_global as ag
import model.m_mysql as db
import app_web as app_web
import common.wky_queues as wqs
#
#from apps.esp.esp_main import EspMain as EspMain
from apps.rgl.rgl_main import RglMain as RglMain
#from apps.gile.udp_server import UdpServer as UdpServer
#from ann.cnn.slim_inresv2 import SlimInresV2 as SlimInresV2

def test1():
    # 试验读取静态方法
    #CQues.test()
    db.init_db_pool()
    CRecommendEngine.test()
    
def test_esp():
    #EspMain.startup({})
    #RglMain.startup({})
    pass
    
if __name__ == '__main__':
    print('starting up...')
    #test1()
    #test_esp()
    # UdpServer.startup('', 8088, 1024)o
    #inres = SlimInresV2()
    #inres.predict('d:/awork/d3.jpg')
    #inres.predict('./work/t001.jpg')
    rgl_main = RglMain()
    rgl_main.startup({})
    
