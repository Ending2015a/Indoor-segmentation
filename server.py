from tools.socket.serverSock import serverSock

import sys
import time
import struct

server = serverSock(name=sys.argv[1])
server.create(port=int(sys.argv[2]))

server.waitForClient()

while True:
    print('wait for task')
    cmd = server.recv()

    (a, b) = struct.unpack('ii', cmd)
    print('get task count {0} to {1}'.format(a, b))

    s = 0
    for i in range(a, b):
        s += i
    
    msg = struct.pack('i', s)

    server.send(msg)


