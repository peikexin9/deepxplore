from utils import preprocess_app
import numpy as np
import os
import random
from configs import bcolors


def training_data_generator(training_apps, feats, malwares, path, batch_size=64):
    # training_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.66))  # 66% for training
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(training_apps):
            apps = training_apps[gen_state: len(training_apps)]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = 0
        else:
            apps = training_apps[gen_state: gen_state + batch_size]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = gen_state + batch_size
        yield np.array(X), np.array(y)


def training_data(train_test_apps, feats, malwares, path):
    training_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.66))  # 50% for training
    xs = []
    ys = []
    for training_app in training_apps:
        if training_app in malwares:
            ys.append(np.array([1, 0]))  # malware
        else:
            ys.append(np.array([0, 1]))  # benign
        xs.append(preprocess_app(training_app, feats, path))
    xs = np.array(xs)
    ys = np.array(ys)
    np.save('training_xs', xs)
    np.save('training_ys', ys)
    return xs, ys


def testing_data_generator(testing_apps, feats, malwares, path, batch_size=64):
    # testing_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.34))
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(testing_apps):
            apps = testing_apps[gen_state: len(testing_apps)]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = 0
        else:
            apps = testing_apps[gen_state: gen_state + batch_size]
            y = []
            for app in apps:
                if app in malwares:
                    y.append(np.array([1, 0]))  # malware
                else:
                    y.append(np.array([0, 1]))  # benign
            X = [preprocess_app(app, feats, path) for app in apps]
            gen_state = gen_state + batch_size
        yield np.array(X), np.array(y)


def testing_data(train_test_apps, feats, malwares, path):
    testing_apps = np.random.choice(train_test_apps, int(len(train_test_apps) * 0.34))  # 34% for testing
    xs = []
    ys = []
    for testing_app in testing_apps:
        if testing_app in malwares:
            ys.append(np.array([1, 0]))  # malware
        else:
            ys.append(np.array([0, 1]))  # benign
        xs.append(preprocess_app(testing_app, feats, path))
    xs = np.array(xs)
    ys = np.array(ys)
    np.save('testing_xs', xs)
    np.save('testing_ys', ys)
    return xs, ys


def load_test_data(batch_size=64, path='./dataset/'):
    malwares = []
    with open(path + 'sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])

    feats = set()
    for filename in os.listdir(path + 'feature_vectors'):
        with open(path + 'feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))
    print('feature read finished')

    feats = np.array(list(feats))
    train_test_apps = os.listdir(path + 'feature_vectors')  # 129013 samples
    xs, _ = testing_data(train_test_apps, feats, malwares, path)
    print('reading raw data finished')
    return feats, xs


def load_data(batch_size=64, load=True, path='./dataset/'):
    malwares = []
    with open(path + 'sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])

    feats = set()
    for filename in os.listdir(path + 'feature_vectors'):
        with open(path + 'feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))

    feats = np.array(list(feats))
    if not load:  # read raw data
        train_test_apps = os.listdir(path + 'feature_vectors')  # 129013 samples
        random.shuffle(train_test_apps)
        train_generator = training_data_generator(train_test_apps[:int(len(train_test_apps) * 0.66)], feats, malwares,
                                                  path,
                                                  batch_size=batch_size)
        test_generator = testing_data_generator(train_test_apps[int(len(train_test_apps) * 0.66):], feats, malwares,
                                                path,
                                                batch_size=batch_size)
        # 		training_xs, training_ys = training_data(train_test_apps, feats, malwares, path)
        # 		testing_xs, testing_ys = testing_data(train_test_apps, feats, malwares, path)
        print('reading raw data finished')
    else:
        training_xs = np.load('training_xs')
        training_ys = np.load('training_ys')
        testing_xs = np.load('testing_xs')
        testing_ys = np.load('testing_ys')

    print(bcolors.OKBLUE + 'data loaded' + bcolors.ENDC)
    return feats, int(len(train_test_apps) * 0.1), int(len(train_test_apps) * 0.1), train_generator, test_generator
