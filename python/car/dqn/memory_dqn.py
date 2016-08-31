import os
import random
import numpy as np
import h5py

class ReplayMemory:
    def __init__(self, filepath):
        self.filepath = filepath
        self.memory_size = 10000
        self.dims = (32, 32)
        self.history_length = 4
        self.batch_size = 32
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
        if os.path.isfile(self.filepath):
            self.load()
        else:
            self.actions = np.empty(self.memory_size, dtype = np.uint8)
            self.rewards = np.empty(self.memory_size, dtype = np.integer)
            self.images = np.empty((self.memory_size, 32, 32), dtype = np.uint8)
            self.terminals = np.empty(self.memory_size, dtype = np.bool)
            self.count = 0
            self.current = 0

    def add(self, image, reward, action, terminal):
        assert image.shape == self.dims
        # NB! image is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.images[self.current, ...] = image
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def clear(self):
        self.count, self.current = 0, 0

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.images[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.images[indexes, ...]

    def sample(self):
        assert self.count > self.history_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_length, self.count - 1)
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def save(self):
        f = h5py.File(self.filepath, 'w')
        f.attrs['count'] = self.count
        f.attrs['current'] = self.current
        for (name, array) in zip(['actions', 'rewards', 'images', 'terminals'],
            [self.actions, self.rewards, self.images, self.terminals]):
            dset = f.create_dataset(name, array.shape, dtype=array.dtype)
            dset[...] = array

    def load(self):
        f = h5py.File(self.filepath, 'r')
        """
        for (name, array) in zip(['actions', 'rewards', 'images', 'terminals'],
            [self.actions, self.rewards, self.images, self.terminals]):
            array = f[name][...]
            dset = f.create_dataset(name, array.shape, dtype=array.dtype)
            dset[...] = array
        """
        self.actions = f['actions'][...]
        self.rewards = f['rewards'][...]
        self.images = f['images'][...]
        self.terminals = f['terminals'][...]
        self.count = f.attrs['count']
        self.current = f.attrs['current']

if __name__ == '__main__':
    memory = ReplayMemory('car/replay_memory.h5')
    for i in range(32):
        memory.add(np.ones((32, 32), dtype=np.uint8), 10, 2, False)
    prestates, actions, rewards, poststates, terminals = memory.sample()
    memory.save()
