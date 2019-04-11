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

width = 32
height = 32
# Depth is the number of color in an image. Set to 3 for full color or 1 for grayscale images 
depth = 3 
# Number of Epochs to train, 
num_iter = 5000
# The batch size for training. 
batch_size = 16
input_shape = (width, height, depth)

is_new_star = {"newtarget": 1, "isstar": 1, "asteroid": 1, "isnova": 1, "known": 1, "noise": 0, "ghost": 0, "pity": 0,}

train_df = pd.read_csv("data/af2019-cv-training-20190312/list.csv")

def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])
 
def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)
 
def l2_norm(x):
    return K.l2_normalize(x, axis=-1)

from keras import backend as K
import tensorflow as tf
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def cnn_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape = input_shape,kernel_initializer='uniform'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3),kernel_initializer='uniform'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),kernel_initializer='uniform'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(256,activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                # loss=[focal_loss(alpha=.25, gamma=2)],
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()
    return model

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
    
    output_layer = Dense(1, activation='sigmoid')(l2_norm_out)
    model = Model(model.input, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def dnn_model():
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
    model.add(Dense(256,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='adam')
    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE
    model.summary()
    
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
    
    output_layer = Dense(1, activation='sigmoid')(l2_norm_out)
    model = Model(model_dnn.input, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x

def resnet34_model():
    inpt = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    #(56,56,64)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    #(28,28,128)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    #(14,14,256)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    #(7,7,512)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    # x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1,activation='sigmoid')(x)
 
    model = Model(inputs=inpt,outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

def vgg16_model():
    ResNet50_net = inception_v3.InceptionV3(weights='imagenet', 
                  include_top=False,
                  input_shape=(139,139,3))
    ResNet50_net.trainable = False
    model = Sequential()
    model.add(ResNet50_net)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
    model.summary()
    return model

import random
def prepareImages(data, m, dataset, i=150, j = 150, depth = 1):
    y_train = []
    count = 0
    if depth < 3:
        X_train = np.zeros((m, i, j, 1)) 
    else: 
        X_train = np.zeros((m, i, j, 3))
    for index, row in data.iterrows():
        # if row['judge'] != 'isstar' and row['judge'] != 'pity':
            # continue
        if is_new_star[row['judge']] == 0:
            xx = random.randint(1,5)
            if xx != 1:
                continue
        fig = row['id']
        if (count%500 == 0):
            print("Loading image: ", count+1, ", ", fig)
        y = row['x']
        x = row['y']
        # if count == 0:
        #     print(x, y)
        folder_name = fig[0:2]
        img_a = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_a.jpg")
        img_b = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_b.jpg")
        img_c = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_c.jpg")
        # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        # img = cv2.merge([img_a, img_b, img_c])
        img = img_a
        img.astype(np.float32)
        img = img / 255.0

        height, width, channel = img.shape
        board = 16
        while height < board * 2 or width < board * 2:
            img = cv2.resize(img, (height * 2, width * 2))
            height, width, channel = img.shape
        x = max(board, x)
        y = max(board, y)
        x = min(height - board, x)
        y = min(width - board, y)
        # img = img[x - board : x + board, y - board : y + board]
        img = cv2.resize(img, (i, j))
        # img = img.reshape(i, j, 1)
        X_train[count] = img
        # if row['judge'] != 'known':
        # print(row['judge'])
        # print(x, y)
        # if row['judge'] != 'noise':
        # cv2.imshow('img', img)
        # q = cv2.waitKey(0)
        y_train.append(is_new_star[row['judge']])
        count += 1
    X_train = X_train[0 : count, : , : , : ]
    return X_train, y_train

def prepare_labels(y):
    # Convert the labels from names or numbers to a one-hot encoding. 
    # Not used in this binary example, but very useful for multi-class classification 
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

X, y = prepareImages(train_df, train_df.shape[0], "train", height, width, depth)

augments = ImageDataGenerator(
        featurewise_center=False,  # input mean to 0 over the all the data
        samplewise_center=False,  # sample mean to 0
        featurewise_std_normalization=False,  # divide by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # 
        rotation_range=10,  # random rotation, 0 to 180
        zoom_range = 0.10, # Randomly zoom in 
        width_shift_range=0.10,  # random horizontally, .01 to .99, fraction of width
        height_shift_range=0.10,  # random vertical shift. .01 to .99 fraction of height
        horizontal_flip=True,  # flip images horizontal, random
        rescale= 1/255,
        fill_mode='nearest',
        vertical_flip=True)  # flip images vertical, random 

# augments.fit(X)
steps_epoch = X.shape[0] // batch_size
random_state = 2018
X, X_val, y, y_val = train_test_split(X, y, test_size = 0.2, random_state=random_state)
# print(y_val)
model = cnn_model()
# model.summary()

def scheduler(epoch):
    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.01)
        print("lr changed to {}".format(lr * 0.01))
    # if epoch == 10:
    #     lr = K.get_value(model.optimizer.lr)
    #     K.set_value(model.optimizer.lr, lr * 0.1)
    #     print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
    '''
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 100 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)
	'''

from keras.callbacks import *
reduce_lr = LearningRateScheduler(scheduler)

# history = model.fit_generator(augments.flow(X, y,batch_size=batch_size),  
#         callbacks = [#tensorboard, checkpointer ,
#         # reduce_lr,
#         #wechat_utils.sendmessage(savelog=True,fexten='TEST')
#         ],
#         epochs=num_iter, steps_per_epoch=steps_epoch, validation_data= (X_val,y_val), verbose=1)

history = model.fit(
        x = X,
        y = y,
        batch_size=batch_size,  
        callbacks = [#tensorboard, checkpointer ,
        # reduce_lr,
        #wechat_utils.sendmessage(savelog=True,fexten='TEST')
        ],
        epochs = num_iter,
        validation_data = (X_val, y_val),
        verbose=1
        )