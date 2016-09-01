import os
import random
import numpy as np
import h5py

class ReplayMemory:
    def __init__(self, pre_training):
        self.pre_training = pre_training
        self.memory_size = 1000
        self.batch_size = 32

        if self.pre_training:
            self.push = self.push_label
            self.sample = self.sample_label
            self.load = self.load_label
            self.save = self.save_label
            self._camera = np.empty((self.batch_size, 3, 224, 224), dtype=np.uint8)
            self.filepath = 'car/dqn/memory_label.h5'
        else:
            self.push = self.push_dqn
            self.sample = self.sample_dqn
            self.load = self.load_dqn
            self.save = self.save_dqn
            self._pre_camera = np.empty((self.batch_size, 3, 224, 224), dtype=np.uint8)
            self._post_camera = np.empty((self.batch_size, 3, 224, 224), dtype=np.uint8)
            self.filepath = 'car/dqn/memory_dqn.h5'

        if os.path.isfile(self.filepath):
            self.load()
        else:
            self.camera_inputs = np.empty((self.memory_size, 3, 224, 224), dtype = np.uint8)
            self.count = 0
            self.current = 0
            if self.pre_training:
                self.labels = np.empty(self.memory_size, dtype=np.uint8)
            else:
                self.actions = np.empty(self.memory_size, dtype=np.uint8)
                self.rewards = np.empty(self.memory_size, dtype=np.float32)
                self.terminals = np.empty(self.memory_size, dtype=np.bool)
        print('[Memory] Ready')

    def push_label(self, camera_input, label):
        self.camera_inputs[self.current, ...] = camera_input
        self.labels[self.current] = label
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def push_dqn(self, camera_input, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.camera_inputs[self.current, ...] = camera_input
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def clear(self):
        self.count, self.current = 0, 0

    def sample_label(self):
        indexes = []
        while len(indexes) < self.batch_size:
            index = random.randint(0, self.count-1)
            while index != self.current - 1:
                index = random.randint(0, self.count-1)

            self._camera[len(indexes), ...] = self.camera_inputs[index]
            indexes.append(index)

        labels = self.labels[indexes]

        return self._camera, labels

    def sample_dqn(self):
        indexes = []
        while len(indexes) < self.batch_size:
            index = random.randint(0, self.count-1)
            while index != self.current - 1 and not self.terminals[index]:
                index = random.randint(0, self.count-1)

            self._pre_camera[len(indexes), ...] = self.camera_inputs[index]
            if index == self.memory_size - 1:
                self._post_camera[len(indexes), ...] = self.camera_inputs[0]
            else:
                self._post_camera[len(indexes), ...] = self.camera_inputs[index+1]
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self._pre_camera, actions, rewards, self._post_camera, terminals

    def save_label(self):
        f = h5py.File(self.filepath, 'w')
        f.attrs['count'] = self.count
        f.attrs['current'] = self.current
        for (name, array) in zip(['camera_inputs', 'labels'],
            [self.camera_inputs, self.labels]):
            dset = f.create_dataset(name, array.shape, dtype=array.dtype)
            dset[...] = array

    def save_dqn(self):
        f = h5py.File(self.filepath, 'w')
        f.attrs['count'] = self.count
        f.attrs['current'] = self.current
        for (name, array) in zip(['actions', 'rewards', 'camera_inputs', 'terminals'],
            [self.actions, self.rewards, self.camera_inputs, self.terminals]):
            dset = f.create_dataset(name, array.shape, dtype=array.dtype)
            dset[...] = array

    def load_label(self):
        f = h5py.File(self.filepath, 'r')
        self.camera_inputs = f['camera_inputs'][...]
        self.labels = f['labels'][...]
        self.count = f.attrs['count']
        self.current = f.attrs['current']

    def load_dqn(self):
        f = h5py.File(self.filepath, 'r')
        self.actions = f['actions'][...]
        self.rewards = f['rewards'][...]
        self.camera_inputs = f['camera_inputs'][...]
        self.terminals = f['terminals'][...]
        self.count = f.attrs['count']
        self.current = f.attrs['current']

if __name__ == '__main__':
    memory = ReplayMemory(pre_training=False)
    for i in range(32):
        memory.push(np.ones((3, 224, 224), dtype=np.uint8), 10, 2, True)
    pre_camera, actions, rewards, post_camera, terminals = memory.sample()
    print(terminals)
    memory.save()
    memory1 = ReplayMemory(pre_training=True)
    for i in range(32):
        memory1.push(np.ones((3, 224, 224), dtype=np.uint8), 2)
    camera, labels = memory1.sample()
    print(labels)
    memory1.save()
