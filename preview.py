from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import fbeta_score
from mpl_toolkits.mplot3d import Axes3D

# import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

# from utils import get_train_labels, tags_to_array, get_train_frame


def fbeta(p, r, beta=2):
    beta2 = beta ** 2
    return (1 + beta2) * p * r / (beta2 * p + r)

if __name__ == "__main__":
    # frame = get_train_frame()
    # print(frame.head())
    # labels = get_train_labels(frame)
    #
    # arrays = []
    # for tags in frame['tags'].values:
    #     arrays.append(tags_to_array(tags, labels))
    #
    # x = labels
    # y = np.sum(arrays, axis=0)
    # y, x = zip(*sorted(zip(y, x), reverse=True))
    #
    # y_pos = np.arange(len(x))
    # plt.bar(y_pos, y, align='center', alpha=0.5)
    # plt.xticks(y_pos, x, rotation=45)
    # plt.ylabel('Occurrences')
    # plt.title('Categories')
    # plt.show()
    #
    # plt.close('all')
    #
    # fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    #
    # arrays = np.array(arrays)
    # for i in range(17):
    #     iarray = list(enumerate(arrays[:, i]))
    #     np.random.shuffle(iarray)
    #     for index, y in iarray:
    #         if y > 0:
    #             print(labels[i], index)
    #             img = cv2.imread('../data/train/train_{}.jpg'.format(index))
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #             title = '{} - {}'.format(labels[i], index)
    #
    #             ax[0].imshow(img)
    #             ax[0].set_title(title)
    #
    #             ax[1].imshow(utils.random_transform(img))
    #             ax[1].set_title('aug1')
    #
    #             ax[2].imshow(utils.random_transform(img))
    #             ax[2].set_title('aug2')
    #             fig.savefig('figure-{}.jpg'.format(labels[i]))
    #             break
    #
    ###########################
    p = np.arange(0, 1.05, 0.05)
    r = np.arange(0, 1.05, 0.05)
    f = np.arange(0, 1.05, 0.05)

    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #

    # surf = ax.plot_surface(p, r, f, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, 1, 0.05)
    Y = np.arange(0, 1, 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = fbeta(X, Y)

    print(fbeta(0.1, 0.5))
    print(fbeta(0.5, 0.1))

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, vmin=0, vmax=1)

    # Customize the z axis.
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('F2')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf)
    plt.show()