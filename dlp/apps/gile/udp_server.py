from socket import *
import threading
from time import ctime
import time

class UdpServer(object):
    host = ''
    port = 8088
    buf_size = 1024
    addr = (host, port)
    
    def startup(host='', port='', buf_size=1024):
        UdpServer.host = host
        UdpServer.port = port
        UdpServer.buf_size = buf_size
        UdpServer.addr = (UdpServer.host, UdpServer.port)
        udp_server = socket(AF_INET, SOCK_DGRAM)
        udp_server.bind(UdpServer.addr)
        idx = 1
        send_addr = None
        recv_addr = None
        while None == send_addr or None == recv_addr:
            client_data, client_addr = udp_server.recvfrom(buf_size)
            if 1 == client_data[0]:
                send_addr = client_addr
            else:
                recv_addr = client_addr
        threads = []
        send_thread = threading.Thread(target=UdpServer.process_send_thread, args=(udp_server, send_addr,))
        threads.append(send_thread)
        recv_thread = threading.Thread(target=UdpServer.process_recv_thread, args=(udp_server, recv_addr,))
        threads.append(recv_thread)
        for thd in threads:
            thd.start()
        for thd in threads:
            thd.join()
        udp_server.close()

    @staticmethod
    def process_send_thread(udp_server, send_addr):
        idx = 1
        while True:
            print('waiting for message...')
            client_data, client_addr = udp_server.recvfrom(UdpServer.buf_size)
            req_msg = str(client_data, encoding='iso8859-1')
            print('received from {0}:{1}'.format(client_addr, req_msg))
            #udp_server.sendto(bytes('message{0}'.format(idx), encoding='iso8859-1'), client_addr)
            idx += 1
        
    @staticmethod
    def process_recv_thread(udp_server, recv_addr):
        idx = 1
        while True:
            time.sleep(1)
            print('send message...')
            #client_data, client_addr = udp_server.recvfrom(buf_size)
            #req_msg = str(client_data, encoding='iso8859-1')
            #print('received from {0}:{1}'.format(client_addr, req_msg))
            udp_server.sendto(bytes('server message{0}'.format(idx), encoding='iso8859-1'), recv_addr)
            idx += 1

