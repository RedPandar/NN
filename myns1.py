# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 13:12:52 2017

@author: Alexsandr
"""
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense



def LeNet_build(width, height, depth, classes, weightsPath=None):
#print(width, height, depth)
model = Sequential()
# first set of CONV => RELU => POOL
model.add(Convolution2D(20, 5, 5, border_mode="same",
input_shape=(depth, height, width)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

        # softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))


if weightsPath is not None:
model.load_weights(weightsPath)
return model
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.33)

    # transform the training and testing labels into vectors in the
    # range [0, classes] -- this generates a vector for each label,
    # where the index of the label is set to `1` and all other entries
    # to `0`; in the case of MNIST, there are 10 class labels
    trainLabels = np_utils.to_categorical(trainLabels, 10)
    testLabels = np_utils.to_categorical(testLabels, 10)

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.01)
    model = LeNet_build(width=28, height=28, depth=1, classes=10, weightsPath=None)