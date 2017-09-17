import cv2
import threading as th
import struct

from tools.tasks.taskManager import taskManager
from tools.log.LogWriter import LogWriter
from tools.hqueue import hQueue


devices = [
    ['joe1', '127.0.0.1', 8888],
    ['joe2', '127.0.0.1', 8889],
    ['joe3', '127.0.0.1', 8890]
]

result_queue = hQueue(capacity=30)


def task(a, b):
    reply = taskManager.sendTask(struct.pack('ii', a, b))
    return struct.unpack('i', reply)

def callback(task, value, state):
    if state == 1:
        result_queue.push((task.create_time, task.value))
    else:
        print(str(value))


def main():
    tm = taskManager(num_threads=3, device_list=devices, log='device.log', name='tm')
    tm.addTask(task, callback, 1, 10000)
    tm.addTask(task, callback, 5, 10000)
    tm.addTask(task, callback, 6, 10000)
    tm.addTask(task, callback, 8, 10000)

    tm.waitCompletion()
    tm.closeDevice()
    

if __name__ == '__main__':
    main()
