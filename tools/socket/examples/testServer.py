from ..simpleSock import serverSock

server = serverSock()
server.create(port=8888)
server.waitForClient()

print(server.recv().decode('ascii'))

server.send('Hello2'.encode('ascii'))
