from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import cv2
import random
from six.moves import cPickle


def calcHist(img, bins):
    hist = np.zeros(bins)
    for row in img:
        for pix in row:
            hist[pix] += 1
    return hist


RELOAD = True
batch_size = 32
nb_epoch = 0
data_augmentation = False

## input image dimensions ##
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

## path to data file ##
X_file = 'X_train.pickle'
Hist_file = 'Hist_train.pickle'
AUX_file = 'AUX_train.pickle'
Y_file = 'Y_train.pickle'

## path to the model weights file ##
model_path = 'letter_model.h5'

matrix_90 = cv2.getRotationMatrix2D((img_cols/2, img_rows/2), 90, 1)
matrix_180 = cv2.getRotationMatrix2D((img_cols/2, img_rows/2), 180, 1)
matrix_270 = cv2.getRotationMatrix2D((img_cols/2, img_rows/2), 270, 1)

COLOR_MAP = {"RED" : 0,
             "ORANGE" : 1,
             "YELLOW" : 2,
             "BLUE" : 3,
             "GRASS" : 4,
             "GREEN" : 5,
             "PINK" : 6,
             "SKY" : 7}
color_classes = len(COLOR_MAP)
LETTER_LIST = [("./letters/letter_A", "RED", 0),
               ("./letters/letter_B", "ORANGE", 1),
               ("./letters/letter_C", "YELLOW", 2),
               ("./letters/letter_D", "GRASS", 3),
               ("./letters/letter_F", "SKY", 4),
               ("./letters/letter_G", "BLUE", 5),
               ("./letters/letter_I", "RED", 6),
               ("./letters/letter_J", "ORANGE", 7),
               ("./letters/letter_K", "YELLOW", 8),
               ("./letters/letter_L", "GRASS", 9),
               ("./letters/letter_M", "GREEN", 10),
               ("./letters/letter_N", "SKY", 11),
               ("./letters/letter_O", "BLUE", 12),
               ("./letters/letter_P", "PINK", 13),
               ("./letters/letter_R", "ORANGE", 14),
               ("./letters/letter_S", "YELLOW", 15),
               ("./letters/letter_U", "GREEN", 16),
               ("./letters/letter_V", "SKY", 17),
               ("./letters/letter_W", "BLUE", 18),
               ("./letters/letter_Z", "ORANGE", 19)]
nb_classes = len(LETTER_LIST)
if (not os.path.exists(data_file)) or RELOAD:
    X_train = []
    Hist_train = []
    AUX_train = []
    Y_train = []
    ## load images ##
    for dir, color, y in LETTER_LIST:
        img_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.jpg')]
        for f in img_files:
            img = cv2.resize(cv2.imread(f), (img_cols, img_rows))
            X_train.append(img)
            X_train.append(cv2.warpAffine(img, matrix_90, (img_cols, img_rows)))
            X_train.append(cv2.warpAffine(img, matrix_180, (img_cols, img_rows)))
            X_train.append(cv2.warpAffine(img, matrix_270, (img_cols, img_rows)))
        AUX_train = AUX_train + [COLOR_MAP[color]]*len(img_files)*4
        Y_train = Y_train + [y]*len(img_files)*4
        print(str(len(img_files))+'*4', "images from "+dir)
    ## calc hist ##
    for img in X_train:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0]
        sat = hsv[:,:,1]
        value = hsv[:,:,2]
        hist = np.concatenate([calcHist(hue, 180), calcHist(sat, 256), calcHist(value, 256)])
        Hist_train.append(hist)
    ## data prprocessing ##
    X_train = np.array(X_train, dtype='float32').transpose(0, 3, 1, 2)/255.
    Hist_train = np.array(Hist_train, dtype='float32')/(img_cols*img_rows)
    # convert class vectors to binary class matrices
    AUX_train = np_utils.to_categorical(AUX_train, color_classes)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)

    ## save data ##
    X_f = open(data_file, 'wb')
    X_f = open(data_file, 'wb')
    X_f = open(data_file, 'wb')
    X_f = open(data_file, 'wb')
    cPickle.dump({"X_train" : X_train,
                  "Hist_train" : Hist_train,
                  "AUX_train" : AUX_train,
                  "Y_train" : Y_train}, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
else:
    f = open(data_file, 'rb')
    data = cPickle.load(f)
    X_train = data['X_train']
    Hist_train = data['Hist_train']
    AUX_train = data['AUX_train']
    Y_train = data['Y_train']
    f.close()

## random shuffled data ##
rand_list = range(X_train.shape[0])
random.shuffle(rand_list)
X_train = X_train[rand_list]
Hist_train = Hist_train[rand_list]
AUX_train = AUX_train[rand_list]
Y_train = Y_train[rand_list]
print(X_train.shape[0], 'train samples')
print('X_train shape:', X_train.shape)
print('Hist_train shape:', Hist_train.shape)
print('AUX_train shape:', AUX_train.shape)
print('Y_train shape:', Y_train.shape)


if not os.path.isfile(model_path):
    ## image cnn ##
    input_img = Input(shape=(img_channels, img_rows, img_cols), name='input_img')
    x = Convolution2D(32, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block1_conv1')(input_img)
    x = Convolution2D(32, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(64, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Convolution2D(128, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(128, 3, 3, init='glorot_normal', activation='relu', border_mode='same', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = Flatten(name='flatten')(x)

    ## histogram fully connected layers ##
    input_hist = Input(shape=(180+256+256, ), name='input_hist')
    h = Dense(1024, init='glorot_normal', activation='relu', name='fc1_h')(input_hist)
    h = Dropout(0.8, name='dropout1_h')(h)
    h = Dense(1024, init='glorot_normal', activation='relu', name='fc2_h')(h)
    h = Dropout(0.8, name='dropout2_h')(h)
    h = Dense(1024, init='glorot_normal', activation='relu', name='fc3_h')(h)
    h = Dropout(0.8, name='dropout3_h')(h)
    h = Dense(color_classes, init='glorot_normal', activation='softmax', name='color_prob')(h)

    ## letter predict fully connected layers ##
    x = merge([x, h], mode='concat', concat_axis=1, name='merge_img_hist')
    x = Dense(1024, init='glorot_normal', activation='relu', name='fc1')(x)
    x = Dropout(0.8, name='dropout')(x)
    x = Dense(nb_classes, init='glorot_normal', activation='softmax', name='letter_prob')(x)

    model = Model(input=[input_img, input_hist], output=[x, h])
else:
    model = load_model(model_path)

# let's train the model using SGD + momentum (how original).
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'],
              loss_weights=[0.1, 0.9])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit([X_train, Hist_train], [Y_train, AUX_train],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)
    model.fit_generator(generator=datagen.flow(X_train, Y_train, batch_size=batch_size),
                        callbacks=[checkpointer],
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))

model.save(model_path)
