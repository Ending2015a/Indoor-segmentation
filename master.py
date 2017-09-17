import cv2
import threading as th
import struct
from datetime import datetime

from tools.tasks.taskManager import taskManager
from tools.log.LogWriter import LogWriter
from tools.hqueue import hQueue


devices = [
    ['Dory1', '140.114.75.144', 8888],
    ['Dory2', '140.114.75.144', 8889],
]

result_queue = hQueue(capacity=30)


def task(img):
    img_bytes = cv2.imencode('.jpg', img)
    seg = taskManager.sendTask(img_bytes)
    seg = cv2.imdecode('.jpg', seg)
    return seg 

def callback(task, value, state):
    if state == 1:
        result_queue.push((task.create_time, value))
    else:
        print(str(value))

def main():
    tm = taskManager(num_threads=3, device_list=devices, log='device.log', name='tm')

    cap = cv2.VideoCapture(0)

    cur_time = datetime.now()

    while True:
        ret, frame = cap.read()

        tm.addTask(task, callback, frame)

        if result_queue.isEnpty():
            cv2.waitKey(1000)
            continue
        
        result_queue.discard_lt((cur_time, None))
        cur_time, seg = result_queue.pop()
        
        output = np.hstack((frame, seg))
        
        cv2.imshow('frame', output)

        cv2.waitKey(30)

    tm.waitCompletion()
    tm.closeDevice()
    

if __name__ == '__main__':
    main()
