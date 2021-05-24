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
from tensorflow.keras import __version__
from zf_unet_224_model import *


def gen_random_image(input_shape, output_shape):
    img = np.zeros(input_shape, dtype=np.uint8)
    mask = np.zeros(output_shape, dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0 + 1, 255)
    light_color1 = random.randint(dark_color1 + 1, 255)
    light_color2 = random.randint(dark_color2 + 1, 255)
    center_0 = random.randint(0, 224)
    center_1 = random.randint(0, 224)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask[random.randint(0, *output_shape)], (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(224):
        for j in range(224):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask


def batch_generator(batch_size, input_shape, output_shape):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = gen_random_image(input_shape, output_shape)
            image_list.append(img)
            mask_list.append([mask])

        image_list = np.array(image_list, dtype=np.float32)
        image_list = preprocess_input(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0
        yield image_list, mask_list


def train_unet(input_shape=(224, 224, 3), output_shape=(1,), epochs=200, batch_size=16, model_name='zf_unet_224.h5',
               optimizer='SGD'):
    learning_rate = 0.001
    model = ZF_UNET_224(input_shape=input_shape, output_shape=output_shape)
    if os.path.isfile(model_name):
        model.load_weights(model_name)

    if optimizer == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('zf_unet_224_temp.h5', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size, input_shape, output_shape),
        epochs=epochs,
        steps_per_epoch=200,
        validation_data=batch_generator(batch_size, input_shape, output_shape),
        validation_steps=200,
        verbose=2,
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
