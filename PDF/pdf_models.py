# usage: python pdf_models.py 1 - train the dave-orig model

from __future__ import print_function

import sys

from keras.layers import Activation, Input, Dense
from keras.utils import to_categorical
from mimicus.tools import datasets

from configs import bcolors
from utils import *


def Model1(input_tensor=None, load_weights=False, num_features=135):  # original dave
    if input_tensor is None:
        input_tensor = Input(shape=(num_features,))
    x = Dense(200, input_dim=num_features, activation='relu', name='fc1')(input_tensor)
    x = Dense(200, activation='relu', name='fc2')(x)
    x = Dense(2, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model1.h5')

    # compiling
    m.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


def Model2(input_tensor=None, load_weights=False, num_features=135):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(num_features,))
    x = Dense(200, input_dim=num_features, activation='relu', name='fc1')(input_tensor)
    x = Dense(200, activation='relu', name='fc2')(x)
    x = Dense(200, activation='relu', name='fc3')(x)
    x = Dense(2, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model2.h5')

    # compiling
    m.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


def Model3(input_tensor=None, load_weights=False, num_features=135):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(num_features,))
    x = Dense(200, input_dim=num_features, activation='relu', name='fc1')(input_tensor)
    x = Dense(200, activation='relu', name='fc2')(x)
    x = Dense(200, activation='relu', name='fc3')(x)
    x = Dense(200, activation='relu', name='fc4')(x)
    x = Dense(2, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model3.h5')

    # compiling
    m.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


if __name__ == '__main__':
    # train the model
    batch_size = 64
    nb_epoch = 10
    model_name = sys.argv[1]

    if model_name == '1':
        model = Model1()
        save_model_name = './Model1.h5'
    elif model_name == '2':
        model = Model2()
        save_model_name = './Model2.h5'
    elif model_name == '3':
        model = Model3()
        save_model_name = './Model3.h5'
    else:
        print(bcolors.FAIL + 'invalid model name, must one of 1, 2 or 3' + bcolors.ENDC)

    # the data, shuffled and split between train and test sets
    X_train, y_train, _ = datasets.csv2numpy('./dataset/train.csv')
    X_test, y_test, _ = datasets.csv2numpy('./dataset/test.csv')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)

    # trainig
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=1)

    # save model
    model.save_weights(save_model_name)
