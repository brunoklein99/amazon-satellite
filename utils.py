import random
import cv2

import pandas as pd
import numpy as np
from keras.preprocessing.image import random_rotation

labels_train_filename = '../labels/train_v2.csv'


def get_train_labels(frame):
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = sorted(list(set(flatten([l.split(' ') for l in frame['tags'].values]))))
    # labels = {l: i for i, l in enumerate(labels)}
    return labels


def get_train_frame():
    return pd.read_csv(labels_train_filename)


def tags_to_array(tags, labels):
    array = np.zeros(17)
    for t in tags.split(' '):
        array[labels.index(t)] = 1
    return array


def random_transform(x):
    if random.random() < 0.5:
        x = cv2.flip(x, 0)

    if random.random() < 0.5:
        x = cv2.flip(x, 1)

    x = random_rotation(x, 90, fill_mode='reflect', row_axis=0, col_axis=1, channel_axis=2)

    return x

