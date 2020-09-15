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
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
#from __future__ import print_function
from keras.layers import Dense
#from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

###############################################################################################################
#images, labels = mndata.load_training()
(x_train, y_train), (x_test, y_test) = mnist.load_data()


###############################################################################################################33
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
#y_train = keras.utils.to_categorical(y_train, 10)
#
x_train = x_train.astype('float32')

x_train /= 255

x_train = x_train.reshape(-1, 28,28, 1)




x_test = x_test.astype('float32')

x_test /= 255

x_test = x_test.reshape(-1, 28,28, 1)

#fx_test=x_test.reshape(y_test.shape[0],img_rows*img_cols)


#y_test = keras.utils.to_categorical(y_test, 10)



print("Training set (images) shape: {shape}".format(shape=x_train.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=x_test.shape))


train_X,valid_X,train_ground,valid_ground = train_test_split(x_train,
                                                             x_train, 
                                                             test_size=0.2, 
                                                             random_state=13)
                                                             
 ################################################################################################# 
noise_factor = 0.5
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=valid_X.shape)                                                           
                                                             
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_valid_noisy = np.clip(x_valid_noisy, 0., 1.)                                                            
                                                             

################################################################################################################
batch_size = 256
epochs = 5
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))


def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 16
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 16
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 32


    #decoder

    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2) # 14 x 14 x 16
    up1 = UpSampling2D((2,2))(conv3) # 28 x 28 x 16
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up1) # 28 x 28 x 1
    return decoded



####################################################################################3
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error',  optimizer=RMSprop())
autoencoder.summary()
autoencoder_train = autoencoder.fit(x_train_noisy, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_valid_noisy, valid_ground))

################################################################################################

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

############################################################################################

pred = autoencoder.predict(x_test)
pred_test = pred.reshape(10000,28,28)
x_test = x_test.reshape(10000,28,28)



fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(x_test[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig



fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(pred_test[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig
#################################################################################################
x_test = x_test.reshape(-1,28,28,1)


x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_test_noisy = x_test_noisy.reshape(10000,28,28)
fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(x_test_noisy[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig
x_test_noisy = x_test_noisy.reshape(-1,28,28,1)
pred = autoencoder.predict(x_test_noisy)
pred_test = pred.reshape(10000,28,28)

fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(pred_test[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_test[i]))
  plt.xticks([])
  plt.yticks([])
fig

