from queue import PriorityQueue

class hQueue(object):
    def __init__(self, capacity=0, throw=True):
        self.queue = PriorityQueue(maxsize=capacity)

    def push(self, item):
        if capacity > 0 and self.queue.qsize() >= capacity:
            if throw:
                self.get()
            else:
                return
        self.queue.put(item)

    def pop(self):
        if self.queue.empty():
            return None
        return self.queue.get()

    def discard_lt(self, item):
        while if not self.queue.empty() and self.queue.queue[0] < item:
            self.queue.get()

    def isEmpty(self):
        return self.queue.empty()

