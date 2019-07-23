import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWebKit import *

class SpiderHtmlRender(QWebPage):
    def __init__(self, url, post_data={}):
        self.app = QApplication(sys.argv)
        QWebPage.__init__(self)
        self.loadFinished.connect(self._loadFinished)
        if post_data:
            self.mainFrame().load(QUrl(url), post_data)
        else:
            self.mainFrame().load(QUrl(url))
        self.app.exec_()
        
    def _loadFinished(self, result):
        self.frame = self.mainFrame()
        self.app.quit() 
