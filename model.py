import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, ZeroPadding2D
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import cv2
import os

from keras.applications import *

def cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = input_shape,kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3),kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (3, 3),kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(1024,activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1,activation='sigmoid'))

    # model.compile(loss = 'binary_crossentropy',
    #             optimizer='adam',
    #             metrics=['accuracy'])
    # model.summary()
    return model

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])
 
def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)
 
def l2_norm(x):
    return K.l2_normalize(x, axis=-1)

def bilinear_cnn_model():
    model = cnn_model()

    #for layer in model.layers:
    #    layer.trainable = False

    cnn_out_a = model.layers[-5].output
    cnn_out_shape = model.layers[-5].output_shape
    # print(cnn_out_shape)
    cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2], cnn_out_shape[3]])(cnn_out_a)
    cnn_out_b = cnn_out_a
    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)

    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)

    #flatten = Flatten()(l2_norm_out)
    #dropout_layer = Dropout(0.5)(flatten)
    
    # output_layer = Dense(1, activation='sigmoid')(l2_norm_out)
    model = Model(model.input, l2_norm_out)
    # model.compile(loss='binary_crossentropy', optimizer='adam',
    #               metrics=['accuracy'])
    model.summary()
    return model

def dnn_model(input_shape):
    seed = 2048 
    np.random.seed(seed) 
    model = Sequential()

    model.add(Conv2D(16, (5, 5), strides=1, input_shape = input_shape,activation='relu', padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(16, (5, 5), strides=1, activation='relu', padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (5, 5), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(32, (5, 5), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=1,activation='relu',padding='same',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    # model.add(AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    # model.add(Dense(1,activation='sigmoid'))
    
    # model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    # model.summary()
    
    return model

def bilinear_dnn_model():
    model_dnn = dnn_model()

    #for layer in model_dnn.layers:
    #    layer.trainable = False

    cnn_out_a = model_dnn.layers[-4].output
    cnn_out_shape = model_dnn.layers[-4].output_shape
    #print(cnn_out_shape)
    cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2], cnn_out_shape[3]])(cnn_out_a)
    cnn_out_b = cnn_out_a
    cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
    cnn_out_dot = Reshape([cnn_out_shape[-1]*cnn_out_shape[-1]])(cnn_out_dot)

    sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
    l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)

    #flatten = Flatten()(l2_norm_out)
    #dropout_layer = Dropout(0.5)(flatten)
    
    # output_layer = Dense(1, activation='sigmoid')(l2_norm_out)
    model = Model(model_dnn.input, l2_norm_out)
    # model.compile(loss='binary_crossentropy', optimizer='adam',
    #               metrics=['accuracy'])
    # model.summary()
    return model

def vgg16_model(input_shape):
    vgg = vgg16.VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=input_shape)
    vgg.trainable = False
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #           optimizer='adam', 
    #           metrics=['accuracy'])
    # model.summary()
    return model

def resnet50_model(input_shape):
    resnet = resnet50.ResNet50(weights='imagenet', 
                  include_top=False,
                  input_shape=input_shape)
    resnet.trainable = False
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #           optimizer='adam', 
    #           metrics=['accuracy'])
    # model.summary()
    return model

def xception_model(input_shape):
    resnet = xception.Xception(weights='imagenet', 
                  include_top=False,
                  input_shape=input_shape)
    resnet.trainable = False
    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy',
    #           optimizer='adam', 
    #           metrics=['accuracy'])
    # model.summary()
    return model

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)