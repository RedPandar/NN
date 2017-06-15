# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:52:32 2017

@author: Alexsandr
"""

import numpy
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# Загрузка данных для обучения
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Изменение размера
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
	# Создание модели
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='sigmoid'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='sigmoid'))
	model.add(Dense(50, activation='sigmoid'))
	model.add(Dense(num_classes, activation='sigmoid'))
	# Компиляция модели
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = load_model('my_model.h5')
 # Построение модели
model = larger_model()
# Заполнение модели данными
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# тестирования модели
scores = model.evaluate(X_test, y_test, verbose=0)
print("Вероятность ошибки %.2f%%" % (100-scores[1]*100))

model.save('my_model.h5')
model.save_weights("weights.h5py")