# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:51:49 2017

@author: Alexsandr
"""

# Plot ad hoc CIFAR10 instances
from keras.datasets import mnist
from matplotlib import pyplot
from scipy.misc import toimage
import keras
print(keras.__version__)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()