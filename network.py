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

    # load data
    #TODO: load data
    x_train = 0
    y_train = 0

    x_test = 0
    y_test = 0


    # build network
    model = build_model()

    # compile network
    model.compile(
        loss='catergorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    # print summary
    model.summary()

    # callbacks for network
    modelCheck = ModelCheckpoint('./best_weights.hdf5', monitor='loss', save_best_only=True, save_weights_only=True)

    # fit model
    history = model.fit(x_train, y_train,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        callbacks=[modelCheck])

    # load best model
    model.load_weights('./best_weights.hdf5}')

    # evaluate results
    eval_results = model.evaluate(x_test, y_test,
                                  batch_size=args.batch_size,
                                  verbose=1)

    # predictions from test set
    preds = model.predict(x_test)

if __name__ == '__main__':
    args = args()
    main(args)