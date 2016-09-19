import os
import random
import numpy as np
import pickle

class ReplayMemory:
    def __init__(self):
        self.memory_size = 10000
        self.filepath = 'memory_car.pickle'

        if os.path.isfile(self.filepath):
            self.load()
        else:
            self.q = []
            self.current, self.count = 0, 0
        print('[Memory] Ready ' + str(self.count))

    def push(self, data):
        print('[Memory] ' + str(self.count))
        self.q.append(data)

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def clear(self):
        self.count, self.current = 0, 0

    def sample(self, batch_size):
        ret = []
        for batch_idx in range(batch_size):
            rand_idx = random.randint(0, self.count-1)
            ret.append(self.q[rand_idx])
        return ret

    def save(self):
        pickle.dump([self.q[0: self.count], self.current], open(self.filepath, 'wb'))

    def load(self):
        self.q, self.current = pickle.load(open(self.filepath, 'rb'))
        self.count = len(self.q)

