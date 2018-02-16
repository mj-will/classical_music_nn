#!/usr/bin/env python

from __future__ import print_function, division

import os
import argparse
import shutil

import keras as K
import numpy as np

from scipy.io import wavfile

from keras.layers import Input, Conv1D, MaxPool1D, Dense
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from utils import split_dataset_test_train

np.random.seed(1337)

# class that contains arguments for network
class args:
    def __init__(self):
        self.Nepochs = 100
        self.batch_size = 64
        self.Nclasses = 3


def build_model(args, input_shape):

    model = Sequential()

    model.add(Conv1D(
        input_shape=input_shape,
        filters=32,
        kernel_size=16,
        activation='relu'
    ))

    model.add(MaxPool1D(
        pool_size=4
    ))

    model.add(Conv1D(
        filters=32,
        kernel_size=16,
        activation='relu'
    ))

    model.add(MaxPool1D(
        pool_size=4
    ))

    model.add(Dense(
        units=256
    ))

    model.add(Dense(
        units=args.Nclasses
    ))

    return model


def optimizer():
    return Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


def main(args):


    # path to all data
    all_data = './data/raw_data30'
    # path to training and test data
    train_data = './data/data30/train'
    test_data = './data/data30/test'
    # percentage of data to use for testing
    test_pct = 0.2

    # split data if necessary
    split_dataset_test_train(all_data, train_data, test_data, test_pct)
    
    # load an image to use as a reference
    img = load_img('data/raw_data30/chopin/chopin0.png')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    # get shape of image
    img_shape = x.shape[1:3]

    # augemntation configuration
    datagen = ImageDataGenerator(rescale=1. / 255)

    # generator to read in images from directories
    train_generator = datagen.flow_from_directory(
        './data/data30/train',
        target_size=(img_shape),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    test_generator = datagen.flow_from_directory(
        './data/data30/test',
        target_size=(img_shape),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    # input shape
    input_shape = (None,) + x.shape[1:]

    # build network
    model = build_model(args, input_shape=input_shape)

    # compile network
    model.compile(
        loss='catergorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    # print summary
    model.summary()

    # callbacks for network
    modelCheck = ModelCheckpoint('./best_weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)

    # fit model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator.filenames) // args.batch_size,
        epochs=args.epochs,
        callbacks=[modelCheck])

    # load best model
    model.load_weights('./best_weights.h5')

    # evaluate results
    eval_results = model.evaluate_generator(
        test_generator,
        steps=(len(test_generator.filenames) // args.batch_size)
    )

    # predictions from test set
    preds = model.predict_generator(
        test_generator,
        steps=(len(test_generator.filenames) // args.batch_size)
    )

if __name__ == '__main__':
    args = args()
    main(args)