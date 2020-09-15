# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:07:25 2019

@author: sidistic
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import plot_model
from scipy.io import loadmat
from keras import backend as K
import numpy 
import keras
import scipy
import h5py


img_rows, img_cols = 28, 28

model = keras.models.load_model('/home/sidistic/MNIST_data/dffnn10class.h5')
model.summary()

data = loadmat('/home/sidistic/MNIST_data/mnist_modified_image_test.mat',matlab_compatible='True')
M_image = data['modified_image']


data = loadmat('/home/sidistic/MNIST_data/mnist_image_label_test.mat',matlab_compatible='True')
label = data['label']


############################################################################################################
label = keras.utils.to_categorical(label, 10)
#
M_image = M_image.astype('float32')

M_image /= 255

fM_image=M_image.reshape(label.shape[0],img_rows*img_cols)
#################################################################################################################33

score = model.evaluate(fM_image, label, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])# plotting the metrics