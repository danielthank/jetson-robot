from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from memory import ReplayMemory
import numpy as np
import os
import cv2
import h5py

VGG_MODEL_PATH = 'car/dqn/vgg_model.h5'
TOP_MODEL_PATH = 'car/dqn/top_model.h5'

GAMMA = 0.99 # decay rate of future rewards

def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis=-1)


class DQN:
    def __init__(self, pre_training):
        self.pre_training = pre_training

        if self.pre_training:
            self.train = self.train_label
            self.push = self.push_label
        else:
            self.train = self.train_dqn
            self.push = self.push_dqn

        if not os.path.isfile(TOP_MODEL_PATH):
            top_model_input = Input(shape=vgg.output_shape[1:])
            fc1 = Dense(1024, activation='relu', name='fc1')(top_model_input)
            dropout1 = Dropout(0.5, name='dropout1')(fc1)
            actionQs = Dense(5, name='actionQs')(dropout1)
            top_model = Model(input=[top_model_input], output=[actionQs])
            top_model.save('TOP_MODEL_PATH')
            self.top_model = top_model
        else:
            self.top_model = self.load_model(TOP_MODEL_PATH)

        vgg_model = self.load_model(VGG_MODEL_PATH)
        
        vgg.inbound_nodes[0].input_tensors[0]

        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        self.model = model
        self.save_model()
        else:
            self.model = self.load_model()

        # VGG trainable = False
        self.model.layers[1].trainable=False
        self.model._make_train_function()
        self.model._make_predict_function()
        print(self.model.trainable_weights)

        self.memory = ReplayMemory(self.pre_training)
        print('[Model] ready')

    def push_label(self, image, label):
        image = cv2.resize(image, (224, 224)).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        self.memory.push(image, label)

    def push_dqn(self, image, action, reward, terminal):
        image = cv2.resize(image, (224, 224)).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        self.memory.push(image, action, reward, terminal)

    def train_label(self):
        images, labels = self.memory.sample()
        images = images.astype(np.float32)
        images[:,0,...] -= 103.939
        images[:,1,...] -= 116.779
        images[:,2,...] -= 123.68
        y = np.zeros((len(labels), 5), dtype=np.float32)
        for i, label in enumerate(labels):
            y[i][label] = 1.
        return self.model.train_on_batch(images, y)

    def train_dqn(self):
        pre_camera, actions, rewards, post_camera, terminals = self.memory.sample()
        pre_camera = pre_camera.astype(np.float32)
        pre_camera[:,0,...] -= 103.939
        pre_camera[:,1,...] -= 116.779
        pre_camera[:,2,...] -= 123.68

        post_camera = post_camera.astype(np.float32)
        post_camera[:,0,...] -= 103.939
        post_camera[:,1,...] -= 116.779
        post_camera[:,2,...] -= 123.68

        targets = self.model.predict_on_batch(pre_camera) # get state's actionQs for ref
        actionQs_t1 = self.model.predict_on_batch(post_camera) # one step look ahead

        ## calc targets ##
        batch_targetQs = (1 - np.array(terminals))*GAMMA*np.max(actionQs_t1, axis=1) + np.array(rewards)
        targets[np.arange(pre_camera.shape[0]), actions] = batch_targetQs

        return self.model.train_on_batch(pre_camera, targets)

    def predict(self, image):
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        images[0,...] -= 103.939
        images[1,...] -= 116.779
        images[2,...] -= 123.68
        image = np.expand_dims(image, axis=0)
        return self.model.predict_on_batch(image)

    def load_model(self, filepath):
        from keras.model import load_model
        return load_model(filepath)

    def save_memory(self):
        self.memory.save()

if __name__ == '__main__':
    model = DQN(pre_training=True)
    model.push(np.zeros((224, 224, 3)), 1)
    model.push(np.zeros((224, 224, 3)), 2)
    model.push(np.zeros((224, 224, 3)), 3)
    print(1)
    model.save_memory()
    print(2)
    print(model.train())
    print(3)
    model.save_model()
    model.save_memory()
    """
    model = DQN(pre_training=False)
    model.push(np.zeros((224, 224, 3)), 2, 1, True)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 1, True)
    print(model.train())
    model.save_model()
    model.save_memory()
    print(model.predict(np.zeros((224, 224, 3))))
    """
