from ..simpleSock import clientSock

client = clientSock()

client.connect()

client.send('Hello'.encode('ascii'))

print(client.recv().decode('ascii'))

