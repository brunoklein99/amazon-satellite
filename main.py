import random

import numpy as np
import pandas as pd
import utils
import settings

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.applications import VGG19
from keras.utils import data_utils

import cv2
from keras.preprocessing.image import random_rotation, random_shift, random_zoom
from os.path import isfile

from sklearn.metrics import fbeta_score


def get_train_generator(frame: pd.DataFrame, shuffle=True):
    while True:
        values = frame.values
        if shuffle:
            np.random.shuffle(values)
        for f, tags in frame.values:
            img = cv2.imread('../data/train/{}.jpg'.format(f))
            img = cv2.resize(img, (settings.input_size, settings.input_size))
            img = utils.random_transform(img)

            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[labels.index(t)] = 1
            yield img, targets


def get_valid_generator(frame: pd.DataFrame):
    while True:
        for f, tags in frame.values:
            img = cv2.imread('../data/train/{}.jpg'.format(f))
            img = cv2.resize(img, (settings.input_size, settings.input_size))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[labels.index(t)] = 1
            yield img, targets


def get_test_generator(frame: pd.DataFrame):
    while True:
        for f, tags in frame.values:
            img = cv2.imread('../data/test/{}.jpg'.format(f))
            img = cv2.resize(img, (settings.input_size, settings.input_size))
            img = utils.random_transform(img)
            yield img


def get_batch_generator_train(gen, batch_size, length):
    while True:
        count = length
        while count > 0:
            bsize = batch_size if count > batch_size else count
            batch_x = np.zeros((bsize, settings.input_size, settings.input_size, settings.input_channels))
            batch_y = np.zeros((bsize, 17))
            for i in range(bsize):
                x, y = gen.__next__()
                batch_x[i] = x
                batch_y[i] = y
            yield batch_x, batch_y
            count -= batch_size


def get_batch_generator_test(gen, batch_size, length):
    while True:
        count = length
        while count > 0:
            bsize = batch_size if count > batch_size else count
            batch_x = np.zeros((bsize, settings.input_size, settings.input_size, settings.input_channels))
            for i in range(bsize):
                batch_x[i] = gen.__next__()
            yield batch_x
            count -= batch_size


def get_best_thresholds(p, y):
    class_indexes = range(p.shape[1])
    best_attempts = np.zeros((p.shape[1],))
    for class_index in class_indexes:
        best_fbeta = 0
        attempts = np.arange(0, 1, .05)
        for attempt in attempts:
            y_prob, y_true = p[:, class_index], y[:, class_index]
            y_pred = y_prob > attempt
            if np.any(y_pred):
                fbeta = fbeta_score(y_true, y_pred, beta=2)
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_attempts[class_index] = attempt

    return best_attempts


def get_predictions_using_thresholds(p, thres):
    for i in range(len(p)):
        for j in range(p.shape[1]):
            p[i, j] = True if p[i, j] > thres[j] else False
    return p


def get_predictions(p, y):
    thres = get_best_thresholds(p, y)
    return thres, get_predictions_using_thresholds(p, thres)


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2] * 17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x

model = Sequential()
model.add(BatchNormalization(input_shape=(settings.input_size, settings.input_size, settings.input_channels)))

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(17, activation='sigmoid'))

df_train_data = pd.read_csv('../labels/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))
labels = sorted(labels)

y_valid = []

df_valid = df_train_data[(len(df_train_data) - settings.valid_data_size):]

for f, tags in df_valid.values:
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[labels.index(t)] = 1
    y_valid.append(targets)

y_valid = np.array(y_valid, np.uint8)

df_train = df_train_data[:(len(df_train_data) - settings.valid_data_size)]
df_valid = df_train_data[(len(df_train_data) - settings.valid_data_size):]

df_train_len = len(df_train.values)
df_valid_len = len(df_valid.values)

gen_train = get_train_generator(df_train)
gen_train = get_batch_generator_train(gen_train, settings.batch_size, df_train_len)
stp_train = np.math.ceil(df_train_len / settings.batch_size)

gen_valid = get_valid_generator(df_valid)
gen_valid = get_batch_generator_train(gen_valid, settings.batch_size, df_valid_len)
stp_valid = np.math.ceil(len(df_valid.values) / settings.batch_size)

df_test_data = pd.read_csv('../labels/sample_submission_v2.csv')

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint('weights.h5',
                             save_best_only=True)]

opt = SGD(lr=settings.learning_rate, decay=settings.lr_decay, momentum=0.9)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

weights_file = data_utils.get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',cache_subdir='models')
model.load_weights(weights_file, by_name=True)

weights_file = 'weights.h5'

if isfile(weights_file):
    model.load_weights(weights_file)
else:
    model.fit_generator(generator=gen_train,
                        steps_per_epoch=stp_train,
                        epochs=settings.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=gen_valid,
                        validation_steps=stp_valid,
                        max_q_size=50)
    model.save_weights(weights_file)

gen_valid = get_valid_generator(df_valid)
gen_valid = get_batch_generator_train(gen_valid, settings.batch_size, df_valid_len)

p_valid = model.predict_generator(generator=gen_valid,
                                  steps=stp_valid,
                                  max_q_size=50,
                                  verbose=1)

threshold = optimise_f2_thresholds(y_valid, p_valid)

for i in range(len(p_valid)):
    for j in range(17):
        p_valid[i, j] = p_valid[i, j] > threshold[j]

print('f2 thr  : ', fbeta_score(y_valid, p_valid, beta=2, average='samples'))

y_test = []

gen_test = get_test_generator(df_test_data)
gen_test = get_batch_generator_test(gen_test, settings.batch_size, len(df_test_data.values))
stp_test = np.math.ceil(len(df_test_data.values) / settings.batch_size)
y_prob = model.predict_generator(generator=gen_test,
                                 steps=stp_test,
                                 max_q_size=50,
                                 verbose=1)

y_prob = np.zeros((len(df_test_data.values), 17))
for i in range(10):
    gen_test = get_test_generator(df_test_data)
    gen_test = get_batch_generator_test(gen_test, settings.batch_size, len(df_test_data.values))
    stp_test = np.math.ceil(len(df_test_data.values) / settings.batch_size)
    y_prob += model.predict_generator(generator=gen_test,
                                      steps=stp_test,
                                      max_q_size=50,
                                      verbose=1)
y_prob /= 10

result = []
for i in range(len(y_prob)):
    tags = []
    for j in range(17):
        if y_prob[i, j] > threshold[j]:
            lbl = labels[j]
            tags.append(lbl)
    result.append(' '.join(tags))

df_test_data['tags'] = result
df_test_data.to_csv('submission-thr.csv', index=False)