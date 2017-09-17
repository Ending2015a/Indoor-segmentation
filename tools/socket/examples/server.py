from serverSock import serverSock




import sys

server = serverSock(name=str(argv[2]))
server.create(port=sys.argv[1])

