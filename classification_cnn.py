from util.dataset_io import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from skimage.transform import resize
import random
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf


def train_cnn1(x_train, y_train, x_test, y_test, num_classes, data_augmentation=False):
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()
    # model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64, 64, 1)))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    #                         border_mode='valid'))
    #
    # model.add(Activation('relu'))
    # model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64, 64, 1)))

    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add((Dense(512)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.fit(x_train, y_train,
              batch_size=128, nb_epoch=4, verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    return score[1]


def classify_images(foldername="data/cropped/", num_users=20, num_classes=5):
    """

    :param foldername:
    :param num_users:
    :param num_classes:
    :return:
    """
    k_fold = 5  # Assume 5-fold CV
    num_groups = num_users // k_fold  # Number of groups to split the data into
    mlp_scores = list()
    user_group_map = dict()  # Random mapping of users to groups
    num_users_in_group = [0 for i in
                          range(num_groups)]  # Store the number of users in a group to ensure uniform distribution

    for i in range(0, num_users):
        rand_int = random.randrange(0, num_groups)
        while num_users_in_group[rand_int] >= 5:
            rand_int = random.randrange(0, num_groups)
        user_group_map[i] = rand_int
        num_users_in_group[rand_int] = 1

    print("Loading images...")
    hog_dataset = unpickle_hog_arrays("data/cropped_hog.pkl")
    for t0 in range(0, num_groups):
        print("ITERATION:", t0 + 1, "\n----------")
        x_train = list()
        x_crossval = list()
        for i0 in range(1, num_classes + 1):
            filelist = listdir(foldername + str(i0))
            for filename in filelist:
                hog_image = hog_dataset[filename]
                if user_group_map[int(filename.split('_')[0]) - 1] == t0:
                    x_crossval.append((resize(hog_image, (64, 64)), i0 - 1))
                else:
                    x_train.append((resize(hog_image, (64, 64)), i0 - 1))

        random.shuffle(x_train)
        random.shuffle(x_crossval)
        y_train = np_utils.to_categorical([x[1] for x in x_train], num_classes)
        x_train = np.array([x[0] for x in x_train]).reshape(len(x_train), 64, 64, 1)
        y_crossval = np_utils.to_categorical([x[1] for x in x_crossval], num_classes)
        x_crossval = np.array([x[0] for x in x_crossval]).reshape(len(x_crossval), 64, 64, 1)

        print("Size of Training Set:", len(x_train), "\nSize of Crossval Set:", len(x_crossval))
        print("Training CNN Classifier...")
        test_score = train_cnn1(x_train, y_train, x_crossval, y_crossval, num_classes)
        print("\nIteration Accuracy:", test_score)
        mlp_scores.append(test_score)
        print("\n")

    print("Average CNN Accuracy:", sum(mlp_scores)/num_groups)

classify_images()
