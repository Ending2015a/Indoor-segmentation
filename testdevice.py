from tools.tasks.deviceManager import deviceManager
from tools.tasks.threadpool import ThreadPool


import time
import select

dlist=[
    ['joe1', '127.0.0.1', 8888],
    ['joe2', '127.0.0.1', 8889],
    ['joe3', '127.0.0.1', 8890],
]

pool = ThreadPool(3, log='client.log', name='pool')

dm = deviceManager(dlist, log='client.log')

number = dm.waitForConnections(5)


def task(dm, num):
    msg = 'work {0}'.format(num).encode('ascii')
    reply = dm.sendMsgToIdleDeviceAndWaitForReply(msg, True)
    print(reply.decode('ascii'))
    time.sleep(1)

def bye(dm, name):
    dm.sendMsgToDevice(name, 'bye'.encode('ascii'))


if number == len(dlist):
    print('all device connected')

print('start sending tasks')
time.sleep(2)



for i in range(10):

    print('add task {0}'.format(i))
    pool.add_task(1, task, dm, 8)

    time.sleep(1.5)

print('::::::::wait for all task done')
pool.wait_completion()

dm.sendMsgToDevice('joe2', 'bye'.encode('ascii'))
print('say bye to device \'joe2\'')
dm.closeDevice('joe2')

for i in range(5):

    print('add task {0}'.format(i))
    pool.add_task(1, task, dm, 10)

    time.sleep(1)


print('::::::::wait for all task done')
pool.wait_completion()

dm.sendMsgToDevice('joe1', 'bye'.encode('ascii'))
dm.sendMsgToDevice('joe3', 'bye'.encode('ascii'))
print('say bye to all the devices')
dm.closeDevice('joe1')
dm.closeDevice('joe3')
    
    
