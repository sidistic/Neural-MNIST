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
from keras.layers import Dense
#from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
###############################################################################################################3


# input image dimensions
img_rows, img_cols = 28, 28

cl1, cl2 = 0, 8 #choose two class you want to evaluate
#mndata = MNIST('/home/sidistic/MNIST_data')
###############################################################################################################
#images, labels = mndata.load_training()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
###############################################################################################################33
i, = np.where(y_train == cl1)   #used to separate index information of class 1
j, = np.where(y_train == cl2)   #used to separate index information of class 2

cl1_train=x_train[i,:,:]        #pooled out the data corresponds to class1
cl1_label=y_train[i]            #pooled out the data labels corresponds to class1

cl2_train=x_train[j,:,:]        #pooled out the data corresponds to class2
cl2_label=y_train[j]            #pooled out the data labels corresponds to class2

train_com = np.concatenate((cl1_train,cl2_train),axis=0)  #Marge the class1 and class2 data
train_lab=np.concatenate((cl1_label,cl2_label),axis=0)   #Marge the labels of class1 and class2

[train_sff,train_labs]=shuffle(train_com,train_lab)     # Shuffle the data and label (to properly train the network)

############################################# Plot to shuffled data ###########################
fig = plt.figure()
for i in range(16):
  plt.subplot(4,4,i+1)
  plt.tight_layout()
  plt.imshow(train_sff[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(train_labs[i]))
  plt.xticks([])
  plt.yticks([])
fig
########################################### Change the labels to 0 and 1 ( as dealing with 2 class), for easy conversion of categorial ###############################
np.place(train_labs, train_labs==cl1, [0])
np.place(train_labs, train_labs==cl2, [1])
############################################################################################################
train_labs_cat = keras.utils.to_categorical(train_labs, 2)          # make the output label categorical
#
train_sff = train_sff.astype('float32')

train_sff /= 255

ftrain_sff=train_sff.reshape(train_labs_cat.shape[0],img_rows*img_cols)  # flattern the input data

################################################################################################################


model = Sequential()  #import Sequential NN model 


model.add(Dense(512, input_dim=784, activation='sigmoid'))	 # 1st hidden layer
model.add(Dense(2, activation='softmax'))                   # output layer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # compile the model
model_log=model.fit(ftrain_sff, train_labs_cat, epochs=30, batch_size=200, validation_split=0.10)  #train the model



####################################### for testing do the same as like training ########################################################################


i, = np.where(y_test == cl1)        #
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
############################################### To see the output in the output node of the test samples #####################################################################
inp = model.input
outputs = [model.layers[1].output]
functor = K.function([inp]+ [K.learning_phase()], outputs )
layer_outs = functor([ftest_com, 0.])                   # give the output in the outputlayer

####################################################################################################################
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.summary()   

############################################ see the loss and accuracy with respect to epochs at the time of training ########################
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'evaluation'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'evaluation'], loc='upper right')
plt.tight_layout()
fig










#i, = np.where(y_test == 1)
#j, = np.where(y_test == 8)