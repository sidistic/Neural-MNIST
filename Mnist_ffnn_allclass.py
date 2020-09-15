# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:58:35 2019

@author: sidistic 
"""

import os
import keras
from keras.datasets import mnist
#from mnist import MNIST
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
from sklearn.utils import shuffle

#from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
###############################################################################################################3



# input image dimensions
img_rows, img_cols = 28, 28

#cl1, cl2 = 1, 7
#mndata = MNIST('/home/sidistic/MNIST_data')
###############################################################################################################
#images, labels = mndata.load_training()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
###############################################################################################################33

fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(x_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig


############################################################################################################
y_train = keras.utils.to_categorical(y_train, 10)
#
x_train = x_train.astype('float32')

x_train /= 255

fx_train=x_train.reshape(y_train.shape[0],img_rows*img_cols)

################################################################################################################


model = Sequential()


model.add(Dense(512, input_dim=784, activation='sigmoid'))	
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_log=model.fit(fx_train, y_train, epochs=30, batch_size=200, validation_split=0.10)
#
#
#
################################################################################################################


x_test = x_test.astype('float32')

x_test /= 255

fx_test=x_test.reshape(y_test.shape[0],img_rows*img_cols)


y_test = keras.utils.to_categorical(y_test, 10)

score = model.evaluate(fx_test, y_test, verbose=1)


####################################################################################################################
########################################################################################################################
inp = model.input
#outputs = [model.layers[3].output]
outputs = [model.layers[1].output]
functor = K.function([inp]+ [K.learning_phase()], outputs )
layer_outs = functor([fx_test, 0.])

####################################################################################################################
print('Test loss:', score[0])
print('Test accuracy:', score[1])# plotting the metrics

model.summary()

fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.tight_layout()
fig



#
#
#
#
#
#
#
##i, = np.where(y_test == 1)
##j, = np.where(y_test == 8)