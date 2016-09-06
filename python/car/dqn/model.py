from __future__ import print_function
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from memory import ReplayMemory
import numpy as np
import os
import cv2
import h5py

DQN_PATH = 'car/dqn/dqn_model.h5'
MODEL_PATH = 'car/dqn/complete_model.h5'
VGG_PATH = 'car/dqn/vgg_model.h5'

FRAME_NUM = 3

GAMMA = 0.99 # decay rate of future rewards
TARGET_UPDATE_FREQ = 10000 # frequency of update params of target freezing network

def getActQ(input_list):
    from keras import backend as KB
    return KB.sum(input_list[0]*input_list[1], axis=1, keepdims=True)

def getActQ_outshape(input_shapes):
    shapes = list(input_shapes)
    assert len(shapes) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2
    return (None, 1)

def getTargetQ(input_list):
    # input list: terminals, target-actionQs, rewards #
    from keras import backend as KB
    return (1 - input_list[0])*GAMMA*KB.max(input_list[1], axis=1, keepdims=True) + input_list[2]

def getTargetQ_outshape(input_shapes):
    shapes = list(input_shapes)
    assert len(shapes) == 3 and len(shapes[0]) == 2 and len(shapes[1]) == 2
    return (None, 1)

def getDQNCost(input_list):
    from keras import backend as KB
    return input_list[0] - input_list[1]

def getDQNCost_outshape(input_shapes):
    shapes = list(input_shapes)
    assert len(shapes) == 2 and len(shapes[0]) == 2 and len(shapes[1]) == 2
    return (None, 1)


class DQN:
    def __init__(self, pre_training):
        self.pre_training = pre_training

        if self.pre_training:
            self.train = self.train_label
            self.push = self.push_label
        else:
            self.train = self.train_dqn
            self.push = self.push_dqn

        if not os.path.isfile(DQN_PATH):
            ## load pre-trained vgg16 network ## 
            vgg = load_model(VGG_PATH)

            ## construct parallel input layers from vgg16 ##
            input_model_template = Model(input=vgg.input, output=vgg.get_layer('maxpooling2d_2').output)
            input_model_outs = []
            input_model_ins = []
            for frame in FRAME_NUM:
                input_model = Model.from_config(input_model_template.get_config())
                input_model.set_weights(input_model_template.get_weights())
                for layer in input_model.layers:
                    if frame == 0:
                        layer.name = layer.name + '_t'
                    else:
                        layer.name = layer.name + '_tm'+str(frame)
                    layer.trainable = False
                input_model_ins.append(input_model.input)
                input_model_outs.append(input_model.output)
            input_merge = merge(input_model_outs, mode='concat', concat_axis=1, name='input_merge')

            ## construct convolution block3 ##
            x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(input_merge)
            x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

            ## construct convolution block4 ##
            x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

            ## construct convolution block5 ##
            x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

            ## Q-values block ##
            x = Flatten(name='flatten')(x)
            x = Dense(1024, activation='relu', name='fc1')(x)
            x = Dropout(0.5, name='dropout1')(x)
            actionQs = Dense(5, name='actionQs')(x)

            ## DQN model ##
            DQN = Model(input=input_model_ins, output=[actionQs])
            DQN.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
            DQN._make_train_function()
            DQN._make_predict_function()
            self.DQN = DQN
            self.save_DQN()
        else:
            self.DQN = self.load_DQN()


        if not self.pre_training:
            ## state-action Q-value ##
            action_in = Input((5, ), dtype='float32', name='action_input')
            actQ = Lambda(getActQ, output_shape=getActQ_outshape, name='action-Q')([self.DQN.output, action_in])

            ## target DQN model ##
            self.target_DQN = Model.from_config(self.DQN.get_config())
            for layer in self.target_DQN.layers:
                layer.name = layer.name + 'target'
                layer.trainable = False
            self.UpdateTarget()

            ## target state-action Q-value ##
            terminals = Input((1, ), dtype='float32', name='terminal')
            rewards = Input((1, ), dtype='float32', name='rewards')
            targetQ_inlist = [terminals, self.target_DQN.output, rewards]
            targetQ = Lambda(getTargetQ, output_shape=getTargetQ_outshape, name='target-Q')(targetQ_inlist)

            ## dqn cost block ##
            cost = Lambda(getDQNCost, output_shape=getDQNCost_outshape, name='dqn-cost')([actQ, targetQ])

            ## whole structure ##
            complete_inputs = complete_inputs + \
                              self.DQN.input + \
                              self.target_DQN.input + \
                              [action_in, terminals, rewards]
            self.complete_model = Model(input=complete_inputs, output=[cost])




        # print(self.model.trainable_weights)

        self.memory = ReplayMemory(self.pre_training)
        print('[Model] ready')

    def UpdateTarget(self):
        self.target_DQN.set_weights(self.DQN.get_weights())

    def push_label(self, image, label):
        image = cv2.resize(image, (224, 224)).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        self.memory.push(image, label)

    def push_dqn(self, img, action, reward, terminal):
        image = cv2.resize(image, (224, 224)).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        self.memory.push(img, action, reward, terminal)

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
        images, actions, rewards, post_camera, terminals = self.memory.sample()
        images = images.astype(np.float32)
        images[:,0,...] -= 103.939
        images[:,1,...] -= 116.779
        images[:,2,...] -= 123.68

        post_camera[:,0,...] -= 103.939
        post_camera[:,1,...] -= 116.779
        post_camera[:,2,...] -= 123.68

        targets = self.model.predict_on_batch(images) # get state's actionQs for ref
        actionQs_t1 = self.target_model.predict_on_batch(post_camera) # one step look ahead

        ## calc targets ##
        batch_targetQs = (1 - np.array(terminals))*GAMMA*np.max(actionQs_t1, axis=1) + np.array(rewards)
        targets[np.arange(images.shape[0]), actions] = batch_targetQs

        return self.model.train_on_batch(images, targets)

    def predict(self, image):
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        images[0,...] -= 103.939
        images[1,...] -= 116.779
        images[2,...] -= 123.68
        image = np.expand_dims(image, axis=0)
        return self.model.predict_on_batch(image)

    def save_DQN(self):
        self.DQN.save(DQN_PATH)

    def load_DQN(self):
        return load_model(DQN_PATH)

    def save_memory(self):
        self.memory.save()

if __name__ == '__main__':
    model = DQN(pre_training=True)
    model.push(np.zeros((224, 224, 3)), 1)
    model.push(np.zeros((224, 224, 3)), 2)
    model.push(np.zeros((224, 224, 3)), 3)
    model.train()
    model.save_model()
    model.save_memory()
    """
    model = DQN(pre_training=False)
    model.push(np.zeros((224, 224, 3)), 2, 1, True)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 1, True)
    model.save()
    """
