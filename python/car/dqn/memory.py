from __future__ import print_function
import os
import random
import numpy as np
import h5py
import cPickle as pickle

from Queue import Queue

class ReplayMemory:
    def __init__(self, pre_training, frame):
        self.pre_training = pre_training
        self.frame = frame
        self.memory_size = 1000
        self.batch_size = 4

        if self.pre_training:
            self.push = self.push_label
            self.sample = self.sample_label
            self.filepath = 'car/dqn/memory_label.pickle'

            self._camera = [np.empty((self.batch_size, 3, 100, 100), dtype=np.uint8)] * self.frame
            self._labels = np.empty((self.batch_size, ), dtype=np.uint8)
        else:
            self.push = self.push_dqn
            self.sample = self.sample_dqn
            self.filepath = 'car/dqn/memory_dqn.pickle'

            self._precamera = [np.empty((self.batch_size, 3, 100, 100), dtype=np.uint8)] * self.frame
            self._postcamera = [np.empty((self.batch_size, 3, 100, 100), dtype=np.uint8)] * self.frame
            self._actions = np.empty((self.batch_size, ), dtype=np.uint8)
            self._rewards = np.empty((self.batch_size, ), dtype=np.float32)
            self._terminals = np.empty((self.batch_size, ), dtype=np.bool)


        if os.path.isfile(self.filepath):
            # print('tfjksdkfhsdf')
            self.load()
        else:
            self.q = []
            self.current, self.count = 0, 0
        print('[Memory] Ready')

    def push_label(self, camera_input, label):
        self.q.append([camera_input, label])

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def push_dqn(self, camera_input, reward, action, terminal):
        self.q.append([camera_input, reward, action, terminal])

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def clear(self):
        self.count, self.current = 0, 0

    def sample_label(self):
        for batch_idx in range(self.batch_size):
            rand_idx = random.randint(0, self.count - self.frame)
            for frame_idx in range(self.frame):
                self._camera[frame_idx][batch_idx, ...] = self.q[rand_idx + frame_idx][0]
            self._labels[batch_idx] = self.q[rand_idx + self.frame - 1][1]

        return self._camera, self._labels

    def sample_dqn(self):
        for batch_idx in range(self.batch_size):
            while True:
                rand_idx = random.randint(0, self.count - self.frame - 1)
                for z in range(self.frame):
                    if self.q[rand_idx + z][3]:
                        break
                if z == self.frame - 1:
                    break
            for frame_idx in range(self.frame):
                self._precamera[frame_idx][batch_idx, ...] = self.q[rand_idx + frame_idx][0]
                self._postcamera[frame_idx][batch_idx, ...] = self.q[rand_idx + frame_idx + 1][0]
            self._actions[batch_idx] = self.q[rand_idx + self.frame][1]
            self._rewards[batch_idx] = self.q[rand_idx + self.frame][2]
            self._terminals[batch_idx] = self.q[rand_idx + self.frame][3]

        return self._precamera, self._actions, self._rewards, self._postcamera, self._terminals

    def save(self):
        pickle.dump([self.q[0: self.count], self.current], open(self.filepath, 'wb'))

    def load(self):
        self.q, self.current = pickle.load(open(self.filepath, 'rb'))
        self.count = len(self.q)
        # print(self.q, self.current, self.count)

if __name__ == '__main__':
    memory = ReplayMemory(pre_training=False, frame=2)
    for i in range(32):
        memory.push(np.ones((3, 100, 100), dtype=np.uint8), 10, 2, False)
    pre_camera, actions, rewards, post_camera, terminals = memory.sample()
    print(terminals)
    memory.save()
    memory1 = ReplayMemory(pre_training=True, frame=2)
    for i in range(32):
        memory1.push(np.ones((3, 100, 100), dtype=np.uint8), 2)
    camera, labels = memory1.sample()
    print(labels)
    memory1.save()
