import os
import random
import numpy as np
import h5py

MEMORY_PATH = 'car/dqn/replay_memory.h5'

class ReplayMemory:
    def __init__(self, pre_training):
        self.pre_training = pre_training

        if self.pre_training:
            self.add = self.add_label
            self.sample = self.sample_label
        else:
            self.add = self.add_dqn
            self.sample = self.sample_dqno

        self.filepath = MEMORY_PATH
        self.memory_size = 10000
        self.batch_size = 32
        self._pre_camera = np.empty((self.batch_size, 3, 224, 224), dtype=np.uint8)
        self._post_camera = np.empty((self.batch_size, 3, 224, 224), dtype=np.uint8)
        if os.path.isfile(self.filepath):
            self.load()
        else:
            self.actions = np.empty(self.memory_size, dtype = np.uint8)
            self.rewards = np.empty(self.memory_size, dtype = np.float32)
            self.camera_inputs = np.empty((self.memory_size, 3, 224, 224), dtype = np.uint8)
            self.terminals = np.empty(self.memory_size, dtype = np.bool)
            self.count = 0
            self.current = 0

    def add_label(self, camera_input, label):
        pass

    def add_dqn(self, camera_input, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.camera_inputs[self.current, ...] = camera_input
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def clear(self):
        self.count, self.current = 0, 0

    def sample_label(self):
        pass

    def sample_dqn(self):
        indexes = []
        while len(indexes) < self.batch_size:
            index = random.randint(1, self.count-1)
            self._pre_camera[len(indexes), ...] = self.camera_inputs[(self.current + index - 1) % self.memory_size]
            self._post_camera[len(indexes), ...] = self.camera_inputs[(self.current + index) % self.memory_size]
            indexes.append(index)
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self._pre_camera, actions, rewards, self._post_camera, terminals

    def save(self):
        f = h5py.File(self.filepath, 'w')
        f.attrs['count'] = self.count
        f.attrs['current'] = self.current
        for (name, array) in zip(['actions', 'rewards', 'camera_inputs', 'terminals'],
            [self.actions, self.rewards, self.camera_inputs, self.terminals]):
            dset = f.create_dataset(name, array.shape, dtype=array.dtype)
            dset[...] = array

    def load(self):
        f = h5py.File(self.filepath, 'r')
        self.actions = f['actions'][...]
        self.rewards = f['rewards'][...]
        self.camera_inputs = f['camera_inputs'][...]
        self.terminals = f['terminals'][...]
        self.count = f.attrs['count']
        self.current = f.attrs['current']

if __name__ == '__main__':
    memory = ReplayMemory()
    for i in range(32):
        memory.add(np.ones((3, 224, 224), dtype=np.uint8), 10, 2, True)
    pre_camera, actions, rewards, post_camera, terminals = memory.sample()
    print(terminals)
    memory.save()
