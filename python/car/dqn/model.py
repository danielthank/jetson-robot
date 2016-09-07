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
VGG_PATH = 'car/dqn/vgg_model.h5'

FRAME_NUM = 1

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
            for frame in xrange(FRAME_NUM):
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
            if FRAME_NUM > 1:
                input_merge = merge(input_model_outs, mode='concat', concat_axis=1, name='input_merge')
            else:
                input_merge = input_model_outs[0]

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
            dqn = Model(input=input_model_ins, output=[actionQs])
            dqn.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
            self.dqn = dqn
            self.save_dqn()
        else:
            self.dqn = self.load_dqn()

        ## build training model ##
        if not self.pre_training:
            ## state-action Q-value ##
            action_in = Input((5, ), dtype='float32', name='action_input')
            actQ = Lambda(getActQ, output_shape=getActQ_outshape, name='action-Q')([self.dqn.output, action_in])

            ## target DQN model ##
            self.target_dqn = Model.from_config(self.dqn.get_config())
            for layer in self.target_dqn.layers:
                layer.name = layer.name + '_target'
                layer.trainable = False
            self.UpdateTarget()

            ## target state-action Q-value ##
            terminals = Input((1, ), dtype='float32', name='terminals')
            rewards = Input((1, ), dtype='float32', name='rewards')
            targetQ_inlist = [terminals, self.target_dqn.output, rewards]
            targetQ = Lambda(getTargetQ, output_shape=getTargetQ_outshape, name='target-Q')(targetQ_inlist)

            ## dqn cost block ##
            cost = Lambda(getDQNCost, output_shape=getDQNCost_outshape, name='dqn-cost')([actQ, targetQ])

            ## whole structure ##
            train_inputs = self.dqn.inputs + \
                           self.target_dqn.inputs + \
                           [action_in, terminals, rewards]
            print(train_inputs)
            self.training_model = Model(input=train_inputs, output=[cost])
            self.training_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
            self.training_model._make_train_function()
            self.training_model._make_predict_function()
        else:
            self.training_model = self.dqn
            self.training_model._make_train_function()
            self.training_model._make_predict_function()

        # print(self.model.trainable_weights)

        self.memory = ReplayMemory(self.pre_training)
        print('[Model] ready')

    def UpdateTarget(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

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
        return self.dqn.train_on_batch(images, y)

    def train_dqn(self):
        ## exprience replay ##
        images, actions, rewards, post_camera, terminals = self.memory.sample()
        batch_size = len(actions)
        ## images preprocessing ##
        images = images.astype(np.float32)
        post_camera = post_camera.astype(np.float32)
        images[:,0,...] -= 103.939
        images[:,1,...] -= 116.779
        images[:,2,...] -= 123.68
        post_camera[:,0,...] -= 103.939
        post_camera[:,1,...] -= 116.779
        post_camera[:,2,...] -= 123.68
        ## actions one-hot encoding ##
        acts_one_hot = np.zeros((batch_size, 5), dtype='float32')
        acts_one_hot[np.arange(batch_size), actions] = 1.
        terminals = np.array(terminals, dtype='float32').reshape((-1,1))
        rewards = np.array(rewards, dtype='float32').reshape((-1,1))

        '''targets = self.model.predict_on_batch(images) # get state's actionQs for ref
        actionQs_t1 = self.target_model.predict_on_batch(post_camera) # one step look ahead

        ## calc targets ##
        batch_targetQs = (1 - np.array(terminals))*GAMMA*np.max(actionQs_t1, axis=1) + np.array(rewards)
        targets[np.arange(images.shape[0]), actions] = batch_targetQs

        return self.model.train_on_batch(images, targets)'''

        train_inputs = [images, post_camera, acts_one_hot, terminals, rewards]
        train_labels = np.zeros((batch_size, 1), dtype='float32')
        return self.training_model.train_on_batch(train_inputs, train_labels)

    def predict(self, image):
        image = cv2.resize(image, (224, 224)).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image[0,...] -= 103.939
        image[1,...] -= 116.779
        image[2,...] -= 123.68
        image = np.expand_dims(image, axis=0)
        return self.dqn.predict_on_batch(image)

    def save_dqn(self):
        self.dqn.save(DQN_PATH)

    def load_dqn(self):
        return load_model(DQN_PATH)

    def save_memory(self):
        self.memory.save()

if __name__ == '__main__':
    """
    model = DQN(pre_training=True)
    #model.dqn.summary()
    #model.training_model.summary()
    #print(model.training_model.trainable_weights)
    print(1)
    model.push(np.zeros((224, 224, 3)), 1)
    model.push(np.zeros((224, 224, 3)), 2)
    model.push(np.zeros((224, 224, 3)), 3)
    print(2)
    model.train()
    print(3)
    model.save_DQN()
    print(4)
    model.save_memory()
    print(5)

    model = DQN(pre_training=False)
    model.push(np.zeros((224, 224, 3)), 2, 1, True)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 2, False)
    model.push(np.zeros((224, 224, 3)), 3, 1, True)
    for i in range(10):
        print(i)
        model.train()
    model.save_DQN()
    model.save_memory()
    """

