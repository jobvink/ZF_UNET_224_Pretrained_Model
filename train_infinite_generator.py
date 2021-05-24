# coding: utf-8
'''
    - train "ZF_UNET_224" CNN with random images
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import argparse
import os
import cv2
import random
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import __version__
from zf_unet_224_model import *





class DataGenerator(Sequence):

    def __init__(self, data_size, batch_size, input_shape, output_shape):
        self.data_size = data_size
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.data = np.zeros((batch_size, *input_shape))
        self.on_epoch_end()

    def gen_random_image(self, input_shape, output_shape):
        img = np.zeros(input_shape, dtype=np.uint8)
        mask = np.zeros((input_shape[0], input_shape[1], *output_shape), dtype=np.uint8)

        # Background
        dark_color = np.zeros(shape=(input_shape[2],))
        for i in range(input_shape[2]):
            dark_color[i] = random.randint(0, 100)
            img[:, :, i] = dark_color[i]

        # Object
        light_color = np.zeros(shape=(input_shape[2],))
        for i in range(input_shape[2]):
            light_color[i] = random.randint(dark_color[i] + 1, 255)

        for c in range(10):
            center_0 = random.randint(0, input_shape[0] - 1)
            center_1 = random.randint(0, input_shape[1] - 1)
            r1 = random.randint(10, 56)
            r2 = random.randint(10, 56)
            img = cv2.ellipse(img.copy(), (center_1, center_0), (r2, r1), 0, 0, 360, light_color, -1)
            ax = random.randint(0, output_shape[0] - 1)
            mask[:, :, ax] = cv2.ellipse(mask[:, :, ax].copy(), (center_1, center_0), (r2, r1), 0, 0, 360, 255, -1)

        noise = np.zeros(input_shape, dtype=np.uint8)
        cv2.randn(noise, 50, 25)
        img += noise
        img = np.clip(img, 0, 255)

        return img, mask

    def __getitem__(self, index):
        X = np.zeros((self.batch_size, *self.input_shape))
        y = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], *self.output_shape))

        for i in range(self.batch_size):
            X[i], y[i] = self.gen_random_image(self.input_shape, self.output_shape)

        return X, y

    def __len__(self):
        return int(np.floor(self.data_size / self.batch_size))


def train_unet(input_shape=(224, 224, 3), output_shape=(1,), epochs=200, batch_size=16, model_name='zf_unet_224.h5',
               optimizer='SGD'):
    learning_rate = 0.001
    model = ZF_UNET_224(input_shape=input_shape, output_shape=output_shape)
    if os.path.isfile(model_name):
        model.load_weights(model_name)

    if optimizer == 'SGD':
        optim = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optim, loss=bce_dice_loss, metrics=[dice_coef])

    model.summary()

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, min_delta=0.00001, verbose=1,
                          mode='min'),
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('zf_unet_224_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    train_gen = DataGenerator(batch_size*100, batch_size, input_shape, output_shape)
    val_gen = DataGenerator(batch_size*10, batch_size, input_shape, output_shape)

    print('Start training...')
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks)

    model.save_weights(model_name)
    pd.DataFrame(history.history).to_csv('zf_unet_224_train.csv', index=False)
    print('Training is finished (weights zf_unet_224.h5 and log zf_unet_224_train.csv are generated )...')


if __name__ == '__main__':
    print('Keras version {}'.format(__version__))

    parser = argparse.ArgumentParser(description='Evaluations of agents for assignment 2')

    parser.add_argument('--input-shape-x', type=int, default=224)
    parser.add_argument('--input-shape-y', type=int, default=224)
    parser.add_argument('--input-channels', type=int, default=3)
    parser.add_argument('--output-channels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-name', type=str, default='zf_unet_224.h5')
    parser.add_argument('--optimizer', type=str, default='SGD')

    args = parser.parse_args()

    train_unet((args.input_shape_x, args.input_shape_y, args.input_channels), (args.output_channels,), args.epochs,
               args.batch_size, args.model_name, args.optimizer)
