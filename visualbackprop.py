import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as k
from keras.applications import VGG19
from keras.layers import Conv2DTranspose, Dense, Flatten, BatchNormalization, MaxPooling2D, Conv2D
from keras.models import model_from_json, Sequential


def get_file_content(filename):
    with open(filename) as f:
        return f.read()


def get_model(filename):
    return model_from_json(get_file_content(filename))


def get_switches(x, pool_size=(2, 2)):
    x = np.copy(x)
    pool_h, pool_w = pool_size
    N, H, W, C = x.shape
    H = int(H / pool_h)
    W = int(W / pool_w)
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    h1 = h * pool_h
                    h2 = h1 + pool_h
                    w1 = w * pool_w
                    w2 = w1 + pool_w
                    window = x[n, h1:h2, w1:w2, c]
                    max = np.max(window)
                    window[window != max] = 0
                    window[window == max] = 1
    return x


def deconv(x, kernel_size=(3, 3), strides=(1, 1)):
    mod = Sequential()
    mod.add(Conv2DTranspose(1, kernel_size=kernel_size, strides=strides, input_shape=x.shape[-3:]))
    w = mod.layers[0].get_weights()
    for i in range(len(w[0])):
        w[0][i] = 1
    for i in range(len(w[1])):
        w[1][i] = 0
    mod.set_weights(w)
    return mod.predict(x)


def shave_padding(x, len):
    _, h, w, _ = x.shape
    h1 = len - 1
    h2 = h1 + h - len
    w1 = len - 1
    w2 = w1 + w - len
    return x[:, h1:h2, w1:w2, :]


def normalize(x):
    max = np.max(x)
    min = np.min(x)
    return (x - min) / (max - min)


def mean(x):
    x = np.mean(x, axis=3)
    return x.reshape(x.shape + (1,))


def upsample(x, kernel_size=(2, 2)):
    h, w = kernel_size
    return np.repeat(x, h, axis=1).repeat(w, axis=2)


def unpool(x, switches):
    x = upsample(x)
    return x * switches


def forward_and_visual(label, mod, img):
    bninp = mod.layers[0].input
    bnout = mod.layers[0].output
    bnfun = k.function([bninp], [bnout])
    bnout = bnfun([img])[0]

    bl1cv1inp = mod.layers[1].input
    bl1cv1out = mod.layers[1].output
    bl1cv1fun = k.function([bl1cv1inp], [bl1cv1out])
    bl1cv1out = bl1cv1fun([bnout])[0]

    bl1cv2inp = mod.layers[2].input
    bl1cv2out = mod.layers[2].output
    bl1cv2fun = k.function([bl1cv2inp], [bl1cv2out])
    bl1cv2out = bl1cv2fun([bl1cv1out])[0]

    bl1mxpinp = mod.layers[3].input
    bl1mxpout = mod.layers[3].output
    bl1mxpfun = k.function([bl1mxpinp], [bl1mxpout])
    bl1mxpout = bl1mxpfun([bl1cv2out])[0]

    ##############################################

    bl2cv1inp = mod.layers[4].input
    bl2cv1out = mod.layers[4].output
    bl2cv1fun = k.function([bl2cv1inp], [bl2cv1out])
    bl2cv1out = bl2cv1fun([bl1mxpout])[0]

    bl2cv2inp = mod.layers[5].input
    bl2cv2out = mod.layers[5].output
    bl2cv2fun = k.function([bl2cv2inp], [bl2cv2out])
    bl2cv2out = bl2cv2fun([bl2cv1out])[0]

    bl2mxpinp = mod.layers[6].input
    bl2mxpout = mod.layers[6].output
    bl2mxpfun = k.function([bl2mxpinp], [bl2mxpout])
    bl2mxpout = bl2mxpfun([bl2cv2out])[0]

    ###############################################

    bl3cv1inp = mod.layers[7].input
    bl3cv1out = mod.layers[7].output
    bl3cv1fun = k.function([bl3cv1inp], [bl3cv1out])
    bl3cv1out = bl3cv1fun([bl2mxpout])[0]

    bl3cv2inp = mod.layers[8].input
    bl3cv2out = mod.layers[8].output
    bl3cv2fun = k.function([bl3cv2inp], [bl3cv2out])
    bl3cv2out = bl3cv2fun([bl3cv1out])[0]

    bl3cv3inp = mod.layers[9].input
    bl3cv3out = mod.layers[9].output
    bl3cv3fun = k.function([bl3cv3inp], [bl3cv3out])
    bl3cv3out = bl3cv3fun([bl3cv2out])[0]

    bl3cv4inp = mod.layers[10].input
    bl3cv4out = mod.layers[10].output
    bl3cv4fun = k.function([bl3cv4inp], [bl3cv4out])
    bl3cv4out = bl3cv4fun([bl3cv3out])[0]

    bl3mxpinp = mod.layers[11].input
    bl3mxpout = mod.layers[11].output
    bl3mxpfun = k.function([bl3mxpinp], [bl3mxpout])
    bl3mxpout = bl3mxpfun([bl3cv1out])[0]

    ###############################################

    bl4cv1inp = mod.layers[12].input
    bl4cv1out = mod.layers[12].output
    bl4cv1fun = k.function([bl4cv1inp], [bl4cv1out])
    bl4cv1out = bl4cv1fun([bl3mxpout])[0]

    bl4cv2inp = mod.layers[13].input
    bl4cv2out = mod.layers[13].output
    bl4cv2fun = k.function([bl4cv2inp], [bl4cv2out])
    bl4cv2out = bl4cv2fun([bl4cv1out])[0]

    bl4cv3inp = mod.layers[14].input
    bl4cv3out = mod.layers[14].output
    bl4cv3fun = k.function([bl4cv3inp], [bl4cv3out])
    bl4cv3out = bl4cv3fun([bl4cv2out])[0]

    bl4cv4inp = mod.layers[15].input
    bl4cv4out = mod.layers[15].output
    bl4cv4fun = k.function([bl4cv4inp], [bl4cv4out])
    bl4cv4out = bl4cv4fun([bl4cv3out])[0]

    bl4mxpinp = mod.layers[16].input
    bl4mxpout = mod.layers[16].output
    bl4mxpfun = k.function([bl4mxpinp], [bl4mxpout])
    bl4mxpout = bl4mxpfun([bl4cv4out])[0]

    ##############################################

    bl5cv1inp = mod.layers[17].input
    bl5cv1out = mod.layers[17].output
    bl5cv1fun = k.function([bl5cv1inp], [bl5cv1out])
    bl5cv1out = bl5cv1fun([bl4mxpout])[0]

    bl5cv2inp = mod.layers[18].input
    bl5cv2out = mod.layers[18].output
    bl5cv2fun = k.function([bl5cv2inp], [bl5cv2out])
    bl5cv2out = bl5cv2fun([bl5cv1out])[0]

    bl5cv3inp = mod.layers[19].input
    bl5cv3out = mod.layers[19].output
    bl5cv3fun = k.function([bl5cv3inp], [bl5cv3out])
    bl5cv3out = bl5cv3fun([bl5cv2out])[0]

    bl5cv4inp = mod.layers[20].input
    bl5cv4out = mod.layers[20].output
    bl5cv4fun = k.function([bl5cv4inp], [bl5cv4out])
    bl5cv4out = bl5cv4fun([bl5cv3out])[0]

    bl5mxpinp = mod.layers[21].input
    bl5mxpout = mod.layers[21].output
    bl5mxpfun = k.function([bl5mxpinp], [bl5mxpout])
    bl5mxpout = bl5mxpfun([bl5cv4out])[0]

    # drop1inp = mod.layers[2].input
    # drop1out = mod.layers[2].output
    # drop1fun = k.function([drop1inp], [drop1out])
    # drop1out = drop1fun([bl5mxpout])[0]
    #
    # flat1inp = mod.layers[3].input
    # flat1out = mod.layers[3].output
    # flat1fun = k.function([flat1inp], [flat1out])
    # flat1out = flat1fun([drop1out])[0]
    #
    # fcinp = mod.layers[4].input
    # fcout = mod.layers[4].output
    # fcfun = k.function([fcinp], [fcout])
    # fcout = fcfun([flat1out])[0]
    #
    # drop2inp = mod.layers[5].input
    # drop2out = mod.layers[5].output
    # drop2fun = k.function([drop2inp], [drop2out])
    # drop2out = drop2fun([fcout])[0]
    #
    # siginp = mod.layers[6].input
    # sigout = mod.layers[6].output
    # sigfun = k.function([siginp], [sigout])
    # sigout = sigfun([drop2out])[0]

    #############################

    bl1cv1out_avg = mean(bl1cv1out)
    bl1cv2out_avg = mean(bl1cv2out)
    bl1cv2out_swi = get_switches(bl1cv2out_avg)

    bl2cv1out_avg = mean(bl2cv1out)
    bl2cv2out_avg = mean(bl2cv2out)
    bl2cv2out_swi = get_switches(bl2cv2out_avg)

    bl3cv1out_avg = mean(bl3cv1out)
    bl3cv2out_avg = mean(bl3cv2out)
    bl3cv3out_avg = mean(bl3cv3out)
    bl3cv4out_avg = mean(bl3cv4out)
    bl3cv4out_swi = get_switches(bl3cv4out_avg)

    bl4cv1out_avg = mean(bl4cv1out)
    bl4cv2out_avg = mean(bl4cv2out)
    bl4cv3out_avg = mean(bl4cv3out)
    bl4cv4out_avg = mean(bl4cv4out)
    bl4cv4out_swi = get_switches(bl4cv4out_avg)

    bl5cv1out_avg = mean(bl5cv1out)
    bl5cv2out_avg = mean(bl5cv2out)
    bl5cv3out_avg = mean(bl5cv3out)
    bl5cv4out_avg = mean(bl5cv4out)
    bl5cv4out_swi = get_switches(bl5cv4out_avg)
    bl5mxpout_avg = mean(bl5mxpout)

    bl5mxpout_up = unpool(bl5mxpout_avg, bl5cv4out_swi)
    bl5mxpout_up = shave_padding(bl5mxpout_up, 2)
    bl5dec4 = deconv(bl5mxpout_up)
    bl5dec4 = bl5dec4 * bl5cv4out_avg
    bl5dec4 = shave_padding(bl5dec4, 2)

    bl5dec3 = deconv(bl5dec4)
    bl5dec3 = bl5dec3 * bl5cv3out_avg
    bl5dec3 = shave_padding(bl5dec3, 2)

    bl5dec2 = deconv(bl5dec3)
    bl5dec2 = bl5dec2 * bl5cv2out_avg
    bl5dec2 = shave_padding(bl5dec2, 2)

    bl5dec1 = deconv(bl5dec2)
    bl5dec1 = bl5dec1 * bl5cv1out_avg

    bl4mxpout_up = unpool(bl5dec1, bl4cv4out_swi)
    bl4mxpout_up = shave_padding(bl4mxpout_up, 2)

    bl4dec4 = deconv(bl4mxpout_up)
    bl4dec4 = bl4dec4 * bl4cv4out_avg
    bl4dec4 = shave_padding(bl4dec4, 2)

    bl4dec3 = deconv(bl4dec4)
    bl4dec3 = bl4dec3 * bl4cv3out_avg
    bl4dec3 = shave_padding(bl4dec3, 2)

    bl4dec2 = deconv(bl4dec3)
    bl4dec2 = bl4dec2 * bl4cv2out_avg
    bl4dec2 = shave_padding(bl4dec2, 2)

    bl4dec1 = deconv(bl4dec2)
    bl4dec1 = bl4dec1 * bl4cv1out_avg

    bl3mxpout_up = unpool(bl4dec1, bl3cv4out_swi)
    bl3mxpout_up = shave_padding(bl3mxpout_up, 2)

    bl3dec4 = deconv(bl3mxpout_up)
    bl3dec4 = bl3dec4 * bl3cv4out_avg
    bl3dec4 = shave_padding(bl3dec4, 2)

    bl3dec3 = deconv(bl3dec4)
    bl3dec3 = bl3dec3 * bl3cv3out_avg
    bl3dec3 = shave_padding(bl3dec3, 2)

    bl3dec2 = deconv(bl3dec3)
    bl3dec2 = bl3dec2 * bl3cv2out_avg
    bl3dec2 = shave_padding(bl3dec2, 2)

    bl3dec1 = deconv(bl3dec2)
    bl3dec1 = bl3dec1 * bl3cv1out_avg

    bl2mxpout_up = unpool(bl3dec1, bl2cv2out_swi)
    bl2mxpout_up = shave_padding(bl2mxpout_up, 2)

    bl2dec2 = deconv(bl2mxpout_up)
    bl2dec2 = bl2dec2 * bl2cv2out_avg
    bl2dec2 = shave_padding(bl2dec2, 2)

    bl2dec1 = deconv(bl2dec2)
    bl2dec1 = bl2dec1 * bl2cv1out_avg

    bl1mxpout_up = unpool(bl2dec1, bl1cv2out_swi)
    bl1mxpout_up = shave_padding(bl1mxpout_up, 2)

    bl1dec2 = deconv(bl1mxpout_up)
    bl1dec2 = bl1dec2 * bl1cv2out_avg
    bl1dec2 = shave_padding(bl1dec2, 2)

    bl1dec1 = deconv(bl1dec2)
    bl1dec1 = bl1dec1 * bl1cv1out_avg

    print(bl1dec1.shape)

    img_map = normalize(bl1dec1[0, :, :, 0])
    img_nor = img[0, :, :, :]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(label)
    ax[0].imshow(img_nor)
    ax[1].imshow(img_map)
    fig.savefig("{}-hetmap".format(label))


    # cv2.imshow('', )
    # cv2.waitKey(0)


if __name__ == "__main__":
    k.set_learning_phase(0)

    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))

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

    model.load_weights('G:\\Source\\planet\\project\\weights.h5')

    images = [
        'train_38350',
        'train_37752',
        'train_40022',
        'train_37865',
        'train_38401',
        'train_39565',
        'train_39821',
        'train_38701',
        'train_35633',
        'train_37533',
        'train_35602',
        'train_39518',
        'train_35795',
        'train_38434',
        'train_39618',
        'train_35588',
        'train_39415',
    ]

    labels = [
        'agriculture',
        'artisinal_mine',
        'bare_ground',
        'blooming',
        'blow_down',
        'clear',
        'cloudy',
        'conventional_mine',
        'cultivation',
        'habitation',
        'haze',
        'partly_cloudy',
        'primary',
        'road',
        'selective_logging',
        'slash_burn',
        'water',
    ]

    for i in range(len(labels)):
        filename = 'G:\\Source\\planet\\data\\train\\{}.jpg'.format(images[i])
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224))
        img = img.reshape((1,) + img.shape)

        forward_and_visual(labels[i], model, img)
