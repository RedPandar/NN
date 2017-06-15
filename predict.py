# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:44:13 2017

@author: Alexsandr
"""
from __future__ import print_function
import numpy as np

import cv2
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model
#from keras.utils.visualize_util import plot
np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_classes = 10
nb_epoch = 5

img_rows, img_cols = 28, 28
nb_filters = 32
nb_pool = 2
nb_conv = 3
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

model = load_model('my_model.h5')
model.load_weights("weights.h5py")
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

out3 = model.predict(X_train[0:6])
print(np.argmax(out3, axis=1))
#print(np.argmax(out3, axis=1),X_train[0:12].shape)
#==============================================================================
# plot(model, to_file='model.png')
#==============================================================================
#img = Image.open('1.png')
#img = img.convert('RGB')
#x = np.asarray(img, dtype='float32')
#x = x.transpose(2, 0, 1)
#x = np.expand_dims(x, axis=0)
#out1 = model.predict(x)
#print(np.argmax(out1))


#,np.argmax(X_train[0:12], axis=2))
#
#img = cv2.imread('1.png')
#x = np.expand_dims(img,axis=0)
#def vis(x):
#     img = np.squeeze(x,axis=0)
#     print(img.shape)
#     plt.imgshot(img)
#features = model.predict(x)