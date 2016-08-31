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

MODEL_PATH = "car/classify/model.h5"

class Model:
    def __init__(self):
        if not os.path.isfile(MODEL_PATH):
            camera_input = Input(shape=(1, 32, 32), dtype='float32', name='camera_input')

            conv1 = Convolution2D(64, 3, 3)(camera_input)
            conv1 = Activation('relu')(conv1)

            conv2 = Convolution2D(64, 3, 3)(conv1)
            conv2 = Activation('relu')(conv2)

            conv3 = Convolution2D(64, 3, 3)(conv2)
            conv3 = Activation('relu')(conv3)

            conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv3 = Dropout(0.25)(conv3)

            conv4 = Convolution2D(128, 3, 3)(conv3)
            conv4 = Activation('relu')(conv4)

            conv5 = Convolution2D(128, 3, 3)(conv4)
            conv5 = Activation('relu')(conv5)

            conv6 = Convolution2D(128, 3, 3)(conv5)
            conv6 = Activation('relu')(conv6)

            conv6 = MaxPooling2D(pool_size=(2, 2))(conv6)
            conv6 = Dropout(0.25)(conv6)
            conv6 = Flatten()(conv6)

            # ir_input = Input(shape=(2,), dtype='uint8', name='ir_input')

            # x = merge([conv6, ir_input], mode='concat')
            x = Dense(500, activation='relu')(conv6)
            x = Dropout(0.5)(x)
            prob = Dense(5, activation='softmax')(x)
            model = Model(input=[camera_input], output=[prob])
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            model._make_train_function()
            model._make_predict_function()
            model.save(MODEL_PATH)
        else:
            model = load_model(MODEL_PATH)

        self.model = model
        self.memory = ReplayMemory('car/model_classify/repaly_memory.h5')
        print('[Model] ready')

    def pre_data(self, img):
        img = np.reshape(img, (1, 1, 32, 32)).astype(np.float32)
        img = img/255
        return img

    def push(self, img, label):
        self.memory.add(self.pre_data(img), label)

    def train(self):
        X, y = self.memory.sample()
        Y = np.zeros((len(y),), dtype=np.float32)
        for i, _y in enumerate(y):
            Y[i][_y] = 1.
        return self.model.train_on_batch(X, Y)

    def predict(self, img):
        img = self.pre_data(img)
        return self.model.predict_on_batch(img)

    def save(self):
        self.model.save(MODEL_PATH)

if __name__ == '__main__':
    model = Model()
    model.train(np.zeros((32, 32)), 3)
    model.save()
