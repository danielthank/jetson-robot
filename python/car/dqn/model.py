from __future__ import print_function
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from memory import ReplayMemory
import numpy as np
import os
import cv2
import h5py

DQN_PATH = 'model.h5'

class DQN:
    def __init__(self, camera_shape, motion_shape):
        self.batch_size = 4
        self.camera_shape = camera_shape
        self.motion_shape = motion_shape

        if not os.path.isfile(DQN_PATH):
            ## load pre-trained vgg16 network ## 
            # vgg = VGG16(include_top=False, weights='imagenet', input_tensor = Input(shape=(3, 100, 100)))

            ## construct parallel input layers from vgg16 ##
            input_tensor = Input(shape=camera_shape)
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(input_tensor)
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
            input_model_template = Model(input=input_tensor, output=x)
            input_model_outs = []
            input_model_ins = []
            for frame in xrange(2):
                input_model = Model.from_config(input_model_template.get_config())
                for layer in input_model.layers:
                    layer.name = layer.name + '_tm'+str(frame)
                input_model_ins.append(input_model.input)
                input_model_outs.append(input_model.output)
            input_merge = merge(input_model_outs, mode='concat', concat_axis=1, name='input_merge')

            ## construct convolution block4 ##
            x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(input_merge)
            x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
            x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
            x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
            x = Flatten(name='camera_flatten')(x)

            motion_input = Input(shape=motion_shape)
            # input_model_ins.append(motion_input)
            y = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='motion_conv1')(motion_input)
            y = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='motion_conv2')(y)
            y = MaxPooling2D((2, 2), strides=(2, 2), name='motion_pool')(y)
            y = Flatten(name='motion_flatten')(y)

            # merge_xy = merge([x, y], mode='concat', concat_axis=1, name='merge_camera_motion')

            ## Q-values block ##
            x = Dense(1024, activation='relu', name='fc1')(x)
            x = Dropout(0.5, name='dropout1')(x)
            actionQs = Dense(5, activation='softmax', name='actionQs')(x)

            ## DQN model ##
            dqn = Model(input=input_model_ins, output=[actionQs])
            self.dqn = dqn
        else:
            self.dqn = self.load_dqn()

        rms = RMSprop(lr=0.0001, clipnorm=1.)
        self.dqn.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
        self.save_dqn()

        self.training_model = self.dqn
        self.training_model._make_train_function()
        self.training_model._make_predict_function()
        # print(self.training_model.summary())

        self.memory = ReplayMemory()
        print('[Model] ready', end='\r\n')

    def push(self, data):
        data[0] = data[0].transpose((2, 0, 1))
        data[1] = data[1].transpose((2, 0, 1))
        self.memory.push(data)

    def train(self):
        data = self.memory.sample(self.batch_size)

        preimgs = np.empty((self.batch_size,) + self.camera_shape)
        nowimgs = np.empty((self.batch_size,) + self.camera_shape)
        motions = np.empty((self.batch_size,) + self.motion_shape)
        labels = np.empty((self.batch_size, 5))

        for batch_idx, batch in enumerate(data):
            preimgs[batch_idx] = batch[0].astype(np.float32) - 128
            nowimgs[batch_idx] = batch[1].astype(np.float32) - 128
            motions[batch_idx] = batch[2]
            labels[batch_idx][batch[3]] = 1.

        # return self.dqn.train_on_batch([preimgs, nowimgs, motions], labels)
        return self.dqn.train_on_batch([preimgs, nowimgs], labels)

    def predict(self, data):
        data[0] = data[0].transpose((2, 0, 1))
        data[1] = data[1].transpose((2, 0, 1))
        preimgs = np.empty((1,) + self.camera_shape)
        nowimgs = np.empty((1,) + self.camera_shape)
        motions = np.empty((1,) + self.motion_shape)

        preimgs[0] = data[0].astype(np.float32) - 128
        nowimgs[0] = data[1].astype(np.float32) - 128
        motions[0] = data[2]

        # return self.dqn.predict_on_batch([preimgs, nowimgs, motions])
        return self.dqn.predict_on_batch([preimgs, nowimgs])

    def save_dqn(self):
        self.dqn.save(DQN_PATH)

    def load_dqn(self):
        return load_model(DQN_PATH)

    def save_memory(self):
        self.memory.save()

if __name__ == '__main__':
    """
    model = DQN(pre_training=True, frame=2)
    #model.dqn.summary()
    #model.training_model.summary()
    #print(model.training_model.trainable_weights)
    print(1)
    model.push(np.zeros((100, 100, 3)), 1)
    model.push(np.zeros((100, 100, 3)), 2)
    model.push(np.zeros((100, 100, 3)), 3)
    print(2)
    model.train()
    print(3)
    model.save_dqn()
    print(4)
    model.save_memory()
    print(5)
    """

    model = DQN(pre_training=True, frame=2, motion_shape=(2, 10, 10))
    print(model.dqn.summary())
    model.push(np.zeros((100, 100, 3)), 1)
    model.push(np.zeros((100, 100, 3)), 2)
    model.push(np.zeros((100, 100, 3)), 3)
    model.push(np.zeros((100, 100, 3)), 4)
    model.save_memory()
    for i in range(10):
        model.train()
    model.save_dqn()

