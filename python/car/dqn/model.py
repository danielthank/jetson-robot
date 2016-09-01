from __future__ import print_function
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from memory import ReplayMemory
import numpy as np
import os
import cv2
import h5py

MODEL_PATH = 'car/dqn/model.h5'
VGG_PATH = 'car/dqn/vgg_model.h5'

def pre_data(self, image):
    image = cv2.resize(image, (224, 224)).astype(np.float32)
    image[:,:,0] -= 103.939
    image[:,:,1] -= 116.779
    image[:,:,2] -= 123.68
    image = im.transpose((2,0,1))
    image = np.expand_dims(im, axis=0)
    return image

class DQN:
    def __init__(self, pre_training):
        self.pre_training = pre_training
        if self.pre_training:
            self.train = self.train_label
            self.push = self.push_label
        else:
            self.train = self.train_dqn
            self.push = self.push_dqn
        if not os.path.isfile(MODEL_PATH):
            vgg = load_model(VGG_PATH)
            vgg.trainable = False
            camera_input = Input(shape=(3, 224, 224), dtype='float32')
            """
            ir_input = Input(shape=(2, ), dtype='float32')
            x = merge([vgg(camera_input), ir_input], mode='concat')
            x = Dense(1024, activation='relu')(x)
            """
            x = Dense(1024, activation='relu')(vgg(camera_input))
            x = Dropout(0.5)(x)
            Q = Dense(5)(x)

            model = Model(input=[camera_input], output=[Q])
            model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
            model._make_train_function()
            model._make_predict_function()
            model.save(MODEL_PATH)
        else:
            model = load_model(MODEL_PATH)
        self.model = model
        self.memory = ReplayMemory(self.pre_training)
        print('[Model] ready')

    def push_label(self, img, label):
        self.memory.add(pre_img(img), label)

    def push_dqn(self, img, action, reward, terminal):
        self.memory.add(pre_img(img), action, reward, terminal)

    def train_label(self):
        images, labels = memory.sample()
        
    def train_dqn(self):
        images, actions, rewards, post_camera, terminals = memory.sample()

    def predict(self, img):
        img = self.pre_data(img)
        return self.model.predict_on_batch(img)

    def save(self):
        self.model.save_dqn(MODEL_PATH)

if __name__ == '__main__':
    model = DQN(pre_training=True)
    model.push(np.zeros((3, 224, 224)), 2, 1, True)
    model.push(np.zeros((3, 224, 224)), 3, 2, False)
    model.push(np.zeros((3, 224, 224)), 3, 2, False)
    model.push(np.zeros((3, 224, 224)), 3, 1, True)
    model.train()
    model.save()
