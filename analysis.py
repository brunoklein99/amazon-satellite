import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


def get_biggest_errors_per_class(p, y):
    diff = y - p
    diff_abs = np.abs(diff)
    indexes = np.argmax(diff_abs, axis=0)
    return indexes, diff[indexes, range(17)]


def get_smallest_erros_per_class(p, y):
    diff = y - p
    diff_abs = np.abs(diff)
    diff_abs = diff_abs * y
    diff_abs[diff_abs == 0] = 1
    indexes = np.argmin(diff_abs, axis=0)
    return indexes, diff[indexes, range(17)]


def plot_biggest_errors(labels, values, p, y):
    indexes, errors = get_biggest_errors_per_class(p, y)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    for i in range(len(indexes)):
        filename = '../data/train/{}.jpg'.format(values[indexes[i]][0])
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        col = i % 2
        ax[col].imshow(img)
        if errors[i] > 0:
            ax[col].set_title(labels[i] + '\nfalse negative - ' + values[indexes[i]][0])
        else:
            ax[col].set_title(labels[i] + '\nfalse positive - ' + values[indexes[i]][0])
        if i % 2 == 1 or i == 16:
            fig.savefig('errors{}.jpg'.format(i))


if __name__ == "__main__":
    frame = pd.read_csv('../labels/train_v2.csv')
    print(frame.head())

    labels = frame['tags'].apply(lambda x: x.split(' '))
    print(labels)
