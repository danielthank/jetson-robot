from __future__ import print_function
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, merge, Convolution2D, MaxPooling2D, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils
from memory import ReplayMemory
import numpy as np
import os
import cv2
import h5py

DQN_PATH = 'car/dqn/dqn_model.h5'
VGG_PATH = 'car/dqn/vgg_model.h5'

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
    def __init__(self, pre_training, frame):
        self.pre_training = pre_training
        self.frame = frame

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
            for frame in xrange(self.frame):
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
            if self.frame > 1:
                input_merge = merge(input_model_outs, mode='concat', concat_axis=1, name='input_merge')
            else:
                input_merge = input_model_outs[0]

            ## construct convolution block3 ##
            x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(input_merge)
            x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

            ## construct convolution block4 ##
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

            ## construct convolution block5 ##
            x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

            ## Q-values block ##
            x = Flatten(name='flatten')(x)
            x = Dense(1024, activation='relu', name='fc1')(x)
            x = Dropout(0.5, name='dropout1')(x)
            actionQs = Dense(5, name='actionQs')(x)

            ## DQN model ##
            dqn = Model(input=input_model_ins, output=[actionQs])
            self.dqn = dqn
        else:
            self.dqn = self.load_dqn()

        rms = RMSprop(lr=0.001)
        self.dqn.compile(optimizer=rms, loss='mean_squared_error', metrics=['accuracy'])
        self.save_dqn()

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
            # print(train_inputs)
            self.training_model = Model(input=train_inputs, output=[cost])
            self.training_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
        else:
            self.training_model = self.dqn

        self.training_model._make_train_function()
        self.training_model._make_predict_function()
        # print(self.training_model.summary())

        self.memory = ReplayMemory(self.pre_training, self.frame)
        print('[Model] ready')

    def UpdateTarget(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def push_label(self, image, label):
        image = cv2.resize(image, (224, 224)).astype(np.uint8).transpose((2, 0, 1))
        self.memory.push(image, label)

    def push_dqn(self, image, action, reward, terminal):
        image = cv2.resize(image, (224, 224)).astype(np.uint8).transpose((2, 0, 1))
        self.memory.push(image, action, reward, terminal)

    def train_label(self):
        images, labels = self.memory.sample()
        # print(images)
        batch_size = len(labels)
        frame_size = len(images)
        for frame_idx in range(frame_size):
            images[frame_idx] = images[frame_idx].astype(np.float32)
            images[frame_idx][:,0,...] -= 103.939
            images[frame_idx][:,1,...] -= 116.779
            images[frame_idx][:,2,...] -= 123.68
        y = np.zeros((batch_size, 5), dtype=np.float32)
        for i, label in enumerate(labels):
            y[i][label] = 100.
        return self.dqn.train_on_batch(images, y)

    def train_dqn(self):
        ## exprience replay ##
        precamera, actions, rewards, postcamera, terminals = self.memory.sample()
        # print(actions, rewards, terminals)
        batch_size = len(actions)
        frame_size = len(precamera)

        ## images preprocessing ##
        for frame_idx in range(frame_size):
            precamera[frame_idx] = precamera[frame_idx].astype(np.float32)
            precamera[frame_idx][:,0,...] -= 103.939
            precamera[frame_idx][:,1,...] -= 116.779
            precamera[frame_idx][:,2,...] -= 123.68

            postcamera[frame_idx] = postcamera[frame_idx].astype(np.float32)
            postcamera[frame_idx][:,0,...] -= 103.939
            postcamera[frame_idx][:,1,...] -= 116.779
            postcamera[frame_idx][:,2,...] -= 123.68

        ## actions one-hot encoding ##
        acts_one_hot = np.zeros((batch_size, 5), dtype='float32')
        acts_one_hot[np.arange(batch_size), actions] = 1.
        terminals = np.array(terminals, dtype='float32').reshape((-1,1))
        rewards = np.array(rewards, dtype='float32').reshape((-1,1))

        '''targets = self.model.predict_on_batch(images) # get state's actionQs for ref
        actionQs_t1 = self.target_model.predict_on_batch(post_camera) # one step look ahead

        ## calc targets ##
        batch_targetQs = (1 - np.array(terminals))*GAMMA*np.max(actionQs_t1, axis=1) + np.array(rewards)
        targets[np.arange(pre_camera.shape[0]), actions] = batch_targetQs

        return self.model.train_on_batch(images, targets)'''

        train_inputs = precamera + postcamera + [acts_one_hot] + [terminals] + [rewards]
        train_labels = np.zeros((batch_size, 1), dtype='float32')
        return self.training_model.train_on_batch(train_inputs, train_labels)

    def predict(self, images):
        for frame_idx in range(self.frame):
            images[frame_idx] = cv2.resize(images[frame_idx], (224, 224)).astype(np.float32).transpose((2, 0, 1))
            images[frame_idx] = images[frame_idx][np.newaxis, :]
            images[frame_idx][:,0,...] -= 103.939
            images[frame_idx][:,1,...] -= 116.779
            images[frame_idx][:,2,...] -= 123.68

        return self.dqn.predict_on_batch(images)

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
    model.push(np.zeros((224, 224, 3)), 1)
    model.push(np.zeros((224, 224, 3)), 2)
    model.push(np.zeros((224, 224, 3)), 3)
    print(2)
    model.train()
    print(3)
    model.save_dqn()
    print(4)
    model.save_memory()
    print(5)
    """

    model = DQN(pre_training=False, frame=2)
    model.push(np.zeros((224, 224, 3)), 3, 4, False)
    model.push(np.zeros((224, 224, 3)), 3, 4, False)
    model.push(np.zeros((224, 224, 3)), 3, 4, False)
    model.push(np.zeros((224, 224, 3)), 3, 4, False)
    model.save_memory()
    for i in range(10):
        model.train()
    model.save_dqn()

