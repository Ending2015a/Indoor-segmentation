import socket
from ..log import LogWriter as logw
import struct

class serverSock(object):
    def __init__(self, family=socket.AF_INET, protocol=socket.SOCK_STREAM, log=None, name=None):
        self.sock = socket.socket(family, protocol)
        if type(log) is str:
            self.log = logw.LogWriter(log, printout=True, name=name)
        else:
            self.log = logw.LogWriter('serverlog.log', printout=True, name=name)
        self.log.open()
        self.created = False

    def create(self, port=8888, capacity=1):
        if self.created:
            self.log.Error('server has been already created')
            return False

        try:
            self.port = port
            self.capacity = capacity
            self.sock.bind(('', port))
            self.sock.listen(capacity)
            self.created = True
            self.log.Log('server created successfully!! port:{0}, capacity:{1}'.format(port, capacity))
        except socket.error as e:
            self._log_error('in serverSock.create', type(e).__name__, e)
            return False
        return True
    
    # send btn, msg
    def send(self, msg):
        try:
            btn = int(len(msg))
            self.client.send(struct.pack('i', btn))
            self.client.send(msg)
            self.log.Log('send {0} bytes to client: {1}'.format(btn, self.client.getAddrFormat()))
        except socket.error as e:
            self._log_error('in serverSock.send', type(e).__name__, e)
            return False
        except TypeError:
            self.log.Error('no client connected')
            return False
        return True

    # recv btn msg
    def recv(self):
        try:
            bt = self.client.recv(4)
            (bt, ) = struct.unpack('i', bt)
            data = self.client.recv(bt)
            self.log.Log('recv {0} bytes from client: {1}'.format(bt, self.client.getAddrFormat()))
        except socket.error as e:
            self._log_error('in serverSock.recv', type(e).__name__, e)
            return None
        except TypeError:
            self.log.Error('no client connected')
            return None
        return data

    def waitForClient(self):
        return self._accept()

    def _accept(self):
        sock, addr = self.sock.accept()
        self.client = Connection(sock, addr)
        self.log.Log('client: {0} connected!'.format(self.client.getAddrFormat()))
        return addr

    def getClientInfo(self):
        return (str(self.client))

    def isCreated(self):
        return self.created

    def close(self):
        try:
            self.client.close()
        except:
            pass
        self.sock.close()
        self.log.Log('Server Closed')
        self.log.close()
        self.created = False
        
    def _log_error(self, msg, etype, e):
        self.log.Error(msg)
        self.log.Error('[errno {0}] {1}: {2}'.format(e.errno, etype, e.strerror))

    def __del__(self):
        self.close()



class Connection(object):
    def __init__(self, conn, addr):
        self.conn = conn
        self.ip, self.port = addr
        self.conn.settimeout(None)
        self.state = True

    def send(self, msg):
        try:
            self.conn.send(msg)
        except socket.error as e:
            self.close()
            raise e
                    

    def recv(self, btn):
        try:
            self.recvData = b''
            tbtn = 0
            while tbtn < btn:
                data = self.conn.recv(btn - tbtn)
                if not data:
                    return self.recvData
                self.recvData += data
                tbtn += len(data)
            return self.recvData

        # if error occurred then close connection
        except socket.error as e:
            self.close()
            raise e
    
    def getAddrFormat(self):
        return '{0}:{1}'.format(self.ip, self.port)

    def getAddr(self):
        return (self.ip, self.port)

    def close(self):
        try:
            self.conn.close()
        except:
            pass
        self.state = False

    def __del__(self):
        self.close()
