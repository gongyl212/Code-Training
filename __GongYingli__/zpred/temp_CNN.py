# -*- coding: utf-8 -*-
"""
  nvidia-smi
	File Name ：    temp_CNN
	Description :   a temporary file to test CNN model
	Author :        Gong Yingli
	Time ：         Tue Aug 27 14:11:15 2019
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.optimizers import SGD, Adadelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.callbacks import  ModelCheckpoint, EarlyStopping
from datetime import datetime
from keras.utils import plot_model


#tf.clip_by_value(t,clip_value_min,clip_value_max)
#K.clip(...)

def sinh(x):
    return tf.sinh(x)
def error2(y_true,y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)
def mean_quartic_error(y_true, y_pred):
    return K.mean(K.square(K.square(y_pred - y_true)), axis=-1)
    #return K.mean(K.square(y_pred - y_true), axis=-1)
def averageError(y_true, y_pred):
    #y_pred = tf.reshape(y_pred,[y_pred.shape[0]])
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def cc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
K.clear_session()

batch_size = 256
num_classes = 1
epochs = 500

# input image dimensions
img_rows, img_cols = 19, 21
winlen = 19
x_train = np.load("../train_data/x_train_winlen_" + str(winlen) + ".npy")
y_train = np.load("../train_data/y_train_winlen_" + str(winlen) + ".npy")
x_test = np.load("../test_data/x_test_winlen_" + str(winlen) + ".npy")
y_test = np.load("../test_data/y_test_winlen_" + str(winlen) + ".npy")
x_valid = np.load("../valid_data/x_valid_winlen_" + str(winlen) + ".npy")
y_valid = np.load("../valid_data/y_valid_winlen_" + str(winlen) + ".npy")

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
y_valid = y_valid.reshape(y_valid.shape[0])

input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_valid.shape[0], 'valid samples')

model = Sequential()
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',
                 data_format=None, dilation_rate=(1,1), activation='tanh', use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, 
                 input_shape=input_shape))
model.add(Conv2D(256, (3,3), activation='tanh', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, data_format=None))
model.add(Conv2D(64, (3,3), activation='tanh', padding='same'))
model.add(ZeroPadding2D((2, 2)))
#model.add(Dropout(0.3))
#model.add(MaxPooling2D(pool_size=(2,2), strides=None, data_format=None))
model.add(Conv2D(64, (3,3), activation='tanh', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, data_format=None))
# How many zeros to add at the beginning and end of the 2 padding dimensions (rows and cols)
#model.add(ZeroPadding2D((3, 3)))
model.add(Conv2D(32, (3,3), activation='tanh'))
model.add(Conv2D(32, (2,2), activation='tanh'))
#model.add(MaxPooling2D(pool_size=(2,2), strides=None, data_format=None))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=64, use_bias=True, activation='tanh', name='Dense_1',
                kernel_initializer='glorot_uniform', bias_initializer='zeros',
                kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                kernel_constraint=None, bias_constraint=None))
model.add(Dense(num_classes))

#lr :float>=0 
#momentum :float>=0 
#decay : float>=0 
#nesterov :Boolean Nesterov
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.01, nesterov=True)
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

model.summary()

model.compile(loss=error2,optimizer=adadelta,metrics=[cc])

checkpoint = ModelCheckpoint(filepath='cnn_test.h5', monitor='loss', 
                             verbose=1, save_best_only='True', 
                             mode='min', period=1)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3)

history = model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint, early_stopping])

predt = model.predict(x_test, verbose=1)
#predv = model.predict(x_valid, verbose=1)
np.savetxt("test_cnn_prediction.txt", predt)
#np.savetxt("validcnn_prediction.txt", predv)
#print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# =============================================================================
#fig = plt.figure()
#axcc = fig.add_subplot(131)
#axcc.plot(history.history['cc'])
#axcc.plot(history.history['val_cc'])
#axcc.set_title('model cc')
#axcc.set_ylabel('cc')
#axcc.set_xlabel('epoch')
#axcc.legend(['train', 'test'], loc='upper left')
#
#axloss = fig.add_subplot(133)
#axloss.plot(history.history['loss'])
#axloss.plot(history.history['val_loss'])
#axloss.set_title('model loss')
#axloss.set_ylabel('loss')
#axloss.set_xlabel('epoch')
#axloss.legend(['train', 'test'], loc='lower left')
#fig.savefig('performance.png')
# =============================================================================

#plot_model(model, to_file='model.png')
