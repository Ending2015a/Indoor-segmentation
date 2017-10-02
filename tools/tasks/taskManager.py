from .threadpool import ThreadPool
from .threadpool import lowLevelTask
from .deviceManager import deviceManager
from ..log.LogWriter import LogWriter
from ..socket.clientSock import clientSock 




class taskManager(object):
    pool = None
    dm = None
    def __init__(self, num_threads, capacity=-1, throw=True, device_list=[], log=None, name=None):
        taskManager.pool = ThreadPool(num_threads, capacity, throw, log)
        taskManager.dm = deviceManager(device_list, log=log)

        taskManager.dm.waitForConnections(5)
    

    def addTask(self, func, callback, *args, **kargs):
        task = taskManager.pool.add_task(3, func, callback, *args, **kargs)


    def waitCompletion(self):
        taskManager.pool.wait_completion()
    
    @staticmethod
    def sendTask(msg):
        reply = taskManager.dm.sendMsgToIdleDeviceAndWaitForReply(msg, True)
        return reply



    
    def closeDevice(self):
        taskManager.dm.closeAllDevices()

