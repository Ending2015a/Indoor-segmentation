from ..log.LogWriter import LogWriter

import sys

py_ver = sys.version_info.major

if py_ver == 2:
    from Queue import PriorityQueue
else:
    from queue import PriorityQueue

import threading as th
from datetime import datetime



###
#   lowLevelTask object
#
#   state: 0(queuing) / 1(done) / 2(exception)
#   back: return value by func
#
class lowLevelTask(object):
    
    def __init__(self, func, call, *args, **kargs):
        self.func = func
        self.args = args
        self.kargs = kargs
        self.state = 0
        self.back = None
        self.callback = call
        self.__create_time = datetime.now()

    def create_time(self):
        return self.__create_time

    def __lt__(self, other):
        return self.__create_time < other.__create_time


###
#   Worker object
#   
#   thread object
#
class Worker(th.Thread):

    ### static variable
    total_num_of_thread = 0

    ###
    #   __init__
    #
    #   create Worker
    #   tasks: a queue object which contains tasks
    #   log: log file path
    #
    #   return: None
    #
    def __init__(self, tasks, log=None):
        th.Thread.__init__(self)
        self.tasks = tasks

        name = 'Worker{0}'.format(Worker.total_num_of_thread)
        Worker.total_num_of_thread += 1
        self.log = LogWriter(log, True, name=name)
        
        self.paused = False
        self.pause_cond = th.Condition(th.Lock())

        # this will stop the thread when main thread is terminated
        self.daemon = True
        self.start()



    ###
    #   run
    #
    #   run tasks
    #
    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()

            _, task = self.tasks.get()
            try:
                task.back = task.func(*task.args, **task.kargs)
                task.state = 1
            except Exception as e:
                self.log.Error(str(e))
                task.back = e
                task.state = 2
            finally:
                if task.callback != None:
                    task.callback(task, task.back, task.state)
                self.tasks.task_done()

    ###
    #   pause
    #
    #   pause thread
    #
    def pause(self):
        self.paused = True
        self.pause_cond.acquire()

    ###
    #   resume
    #
    #   resume thread
    #
    def resume(self):
        self.paused = False
        self.pause_cond.notify()
        self.pause_cond.release()


class ThreadPool:

    ###
    #   __init__
    #
    #   num_threads: total number of threads
    #   capacity: the max capacity of queue, if -1 then default capacity = num_threads * 3
    #   log: log file path
    #   name: ThreadPool ID in log file
    #
    def __init__(self, num_threads, capacity=-1, log=None):
        self.workers=[]
        if capacity == -1:
            capacity = num_threads * 3
        self.tasks = PriorityQueue(capacity)

        for n in range(num_threads):
            
            self.workers.append(Worker(self.tasks, log=log))

    ###
    #   add_task
    #
    #   return: task handle
    #
    def add_task(self, priority, func, callback, *args, **kargs):
        task = lowLevelTask(func, callback, *args, **kargs)
        self.tasks.put((priority, task))
        return task

    ###
    #   wait_completion
    #
    #   wait for all tasks done
    #
    def wait_completion(self):
        self.tasks.join()

