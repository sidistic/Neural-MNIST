# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:17:58 2019

@author: sidistic
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:58:35 2019

@author: sidistic 
"""
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
import h5py

#from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

###############################################################################################################3


# input image dimensions
img_rows, img_cols = 28, 28

cl1, cl2 = 0, 8
#mndata = MNIST('/home/sidistic/MNIST_data')
###############################################################################################################
#images, labels = mndata.load_training()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
###############################################################################################################33
i, = np.where(y_train == cl1)
j, = np.where(y_train == cl2)

cl1_train=x_train[i,:,:]
cl1_label=y_train[i]

cl2_train=x_train[j,:,:]
cl2_label=y_train[j]
train_com = np.concatenate((cl1_train,cl2_train),axis=0)
train_lab=np.concatenate((cl1_label,cl2_label),axis=0)

[train_sff,train_labs]=shuffle(train_com,train_lab)

fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(train_sff[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(train_labs[i]))
  plt.xticks([])
  plt.yticks([])
fig

np.place(train_labs, train_labs==cl1, [0])
np.place(train_labs, train_labs==cl2, [1])
############################################################################################################
train_labs_cat = keras.utils.to_categorical(train_labs, 2)
#
train_sff = train_sff.astype('float32')

train_sff /= 255

ftrain_sff=train_sff.reshape(train_labs_cat.shape[0],img_rows*img_cols)

################################################################################################################
model = Sequential() 
model.add(Dense(1024, input_dim=784, activation='sigmoid'))	
model.add(Dense(512, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))	
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model_log=model.fit(ftrain_sff, train_labs_cat, epochs=30, batch_size=200, validation_split=0.10)



###############################################################################################################


i, = np.where(y_test == cl1)
j, = np.where(y_test == cl2)
cl1_test=x_test[i,:,:]
cl1_label=y_test[i]

cl2_test=x_test[j,:,:]
cl2_label=y_test[j]
test_com = np.concatenate((cl1_test,cl2_test),axis=0)
test_lab=np.concatenate((cl1_label,cl2_label),axis=0)
np.place(test_lab, test_lab==cl1, [0])
np.place(test_lab, test_lab==cl2, [1])


test_com = test_com.astype('float32')

test_com /= 255

ftest_com=test_com.reshape(test_lab.shape[0],img_rows*img_cols)


test_lab_cat = keras.utils.to_categorical(test_lab, 2)

score = model.evaluate(ftest_com, test_lab_cat, verbose=1)
########################################################################################################################
inp = model.input
outputs = [model.layers[3].output]
functor = K.function([inp]+ [K.learning_phase()], outputs )
layer_outs = functor([ftest_com, 0.])

####################################################################################################################

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






#i, = np.where(y_test == 1)
#j, = np.where(y_test == 8)