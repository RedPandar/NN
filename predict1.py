# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:44:13 2017

@author: Alexsandr
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
#import cv2
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D

batch_size = 128
nb_classes = 10
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:500]
y_train = y_train[:500]
X_test = X_test[:50]
y_test = y_test[:50]
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def build_model():
 model = Sequential()
 model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='sigmoid'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Convolution2D(15, 3, 3, activation='sigmoid'))
 model.add(MaxPooling2D(pool_size=(2, 2)))
 model.add(Dropout(0.2))
 model.add(Flatten())
 model.add(Dense(128, activation='sigmoid'))
 model.add(Dense(50, activation='sigmoid'))
 model.add(Dense(nb_classes, activation='sigmoid'))
 model.load_weights("weights.h5py")
 model = load_model('my_model.h5')
 return model


model_3 = build_model()
model_3.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

out3 = model_3.predict(X_train[0:10])
print(np.argmax(out3, axis=1))

