import socket
import threading as th
from ..log import LogWriter as logw
import struct


class clientSock(object):
    def __init__(self, family=socket.AF_INET, protocol=socket.SOCK_STREAM, log=None, name=None):
        self.sock = socket.socket(family, protocol)
        self.isConnected = False
        if type(log) is str:
            self.log = logw.LogWriter(log, printout = True, name=name)
        else:
            self.log = logw.LogWriter('clientlog.log', printout = True, name=name)
        
    def connect(self, ip='127.0.0.1', port=8888, time_out=5):
        self.server_addr = {'ip': ip, 'port':port}
        try:
            self.sock.settimeout(time_out)
            self.sock.connect((ip, port))
            self.sock.settimeout(None)
            self.isConnected = True
        except socket.error as e:
            self._log_socket_error('in clientSock.connect', e)
            return False
        return True

    # send btn msg
    def send(self, msg):
        try:
            btn = int(len(msg))
            self.sock.send(struct.pack('i', btn))
            self.sock.send(msg)
            self.log.Log('send {0} bytes to server: {1}'.format(btn, self.getServerAddr()))
        except socket.error as e:
            self._log_socket_error('in clientSock.send', e)
            
            return False
        return True
    
    # recv btn msg
    def recv(self):
        try:
            bt = self._recv(4)
            (bt, ) = struct.unpack('i', bt)
            data = self._recv(bt)
            self.log.Log('recv {0} bytes from server: {1}'.format(bt, self.getServerAddr()))
        except socket.error as e:
            self._log_socket_error('in clientSock.recv', e)
            return None
        return data
        

    def _recv(self, btn):
        self.recvData = b''
        tbtn = 0
        while tbtn < btn:
            data = self.sock.recv(btn-tbtn)
            if not data:
                return self.recvData
            self.recvData += data
            tbtn += len(data)
        return self.recvData
    
    def getServerAddr(self):
        try:
            addr = '{0}:{1}'.format(self.server_addr['ip'], self.server_addr['port'])
            return addr
        except TypeError as e:
            self.log.Error('no server connected')
            return None
                    


    def close(self):
        try:
            self.sock.close()
        except:
            pass
        self.log.Log('client Closed')
        self.isConnected = False

    def _log_socket_error(self, msg, e):
        self.log.Error(msg)
        self.log.Error('[errno {0}] socket error: {1}'.format(e.errno, e.strerror))

    def __del__(self):
        self.close()



