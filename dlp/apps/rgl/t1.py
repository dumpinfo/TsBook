from PySide.QtCore import QByteArray, QUrl
from PySide.QtGui import QApplication  
from PySide.QtWebKit import QWebView, QWebPage 
from PySide.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply


class Browser(object):

    def __init__(self):
        self.network_manager = QNetworkAccessManager()
        self.network_manager.createRequest = self._create_request
        self.network_manager.finished.connect(self._request_finished)

        self.web_page = QWebPage()
        self.web_page.setNetworkAccessManager(self.network_manager)

        self.web_view = QWebView()
        self.web_view.setPage(self.web_page)

    def _create_request(self, operation, request, data):
        print(data.readAll())
        reply = QNetworkAccessManager.createRequest(self.network_manager,
                                                    operation,
                                                    request,
                                                    data)
        return reply

    def _request_finished(self, reply):
        if not reply.error() == QNetworkReply.NoError:
            # request probably failed
            print(reply.error())
            print(reply.errorString())

    def _make_request(self, url):
        request = QNetworkRequest()
        request.setUrl(QUrl(url))
        return request

    def _urlencode_post_data(self, post_data):
        post_params = QUrl()
        for (key, value) in post_data.items():
            post_params.addQueryItem(key, unicode(value))

        return post_params.encodedQuery()

    def perform(self, url, method='GET', post_data=dict()):
        request = self._make_request(url)

        if method == 'GET':
            self.web_view.load(request)
        else:
            encoded_data = self._urlencode_post_data(post_data)
            request.setRawHeader('Content-Type',
                                 QByteArray('application/x-www-form-urlencoded'))
            self.web_view.load(request,
                               QNetworkAccessManager.PostOperation,
                               encoded_data)

if __name__ == '__main__':
    app = QApplication([])
    browser = Browser()
    browser.perform('http://www.python.org', 'POST', {'test': 'value', 'anothername': 'gfdgfd'})
    app.exec_()

