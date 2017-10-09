# usage: python app_models.py 1 - train the dave-orig model

from __future__ import print_function

import sys

from keras.layers import Activation, Input, Dense

from data_utils import *
from utils import *


def Model1(input_tensor=None, load_weights=False, num_features=545334):  # original dave
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


def Model2(input_tensor=None, load_weights=False, num_features=545334):  # original dave with normal initialization
    if input_tensor is None:
        input_tensor = Input(shape=(num_features,))
    x = Dense(50, input_dim=num_features, activation='relu', name='fc1')(input_tensor)
    x = Dense(50, activation='relu', name='fc2')(x)
    x = Dense(2, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    m = Model(input_tensor, x)
    if load_weights:
        m.load_weights('./Model2.h5')

    # compiling
    m.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    print(bcolors.OKGREEN + 'Model compiled' + bcolors.ENDC)
    return m


def Model3(input_tensor=None, load_weights=False, num_features=545334):  # simplified dave
    if input_tensor is None:
        input_tensor = Input(shape=(num_features,))
    x = Dense(200, input_dim=num_features, activation='relu', name='fc1')(input_tensor)
    x = Dense(10, activation='relu', name='fc2')(x)
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

    # the data, shuffled and split between train and test sets
    feats, nb_train, nb_test, train_generator, test_grnerator = load_data(batch_size, False)
    if model_name == '1':
        model = Model1(num_features=len(feats))
        save_model_name = './Model1.h5'
    elif model_name == '2':
        model = Model2(num_features=len(feats))
        save_model_name = './Model2.h5'
    elif model_name == '3':
        model = Model3(num_features=len(feats))
        save_model_name = './Model3.h5'
    else:
        print(bcolors.FAIL + 'invalid model name, must one of 1, 2 or 3' + bcolors.ENDC)

    # trainig
    model.fit_generator(train_generator, steps_per_epoch=nb_train // batch_size, epochs=nb_epoch, workers=8,
                        use_multiprocessing=True)

    # save model
    model.save_weights(save_model_name)

    # evaluate the model
    score = model.evaluate_generator(test_grnerator, steps=nb_test // batch_size)
    print('\n')
    print('Overall Test score:', score[0])
    print('Overall Test accuracy:', score[1])
