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
from model import *
import cv2
import os

from keras.applications import *

width = 32
height = 32
# Depth is the number of color in an image. Set to 3 for full color or 1 for grayscale images 
depth = 3
# Number of Epochs to train, 
num_iter = 1000
# The batch size for training. 
batch_size = 128
input_shape = (width, height, depth)
num_classes = 2

is_new_star = {"newtarget": 1, "isstar": 1, "asteroid": 1, "isnova": 1, "known": 1, "noise": 0, "ghost": 0, "pity": 0,}

train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")

def create_base_network():
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(1024, activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Dense(1024, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred): # numpy上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred): # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def crop_img(img, x, y, i, j):
    height, width, channel = img.shape
    board = 4
    while height < board * 2 or width < board * 2:
        img = cv2.resize(img, (height * 2, width * 2))
        height, width, channel = img.shape
    x = max(board, x)
    y = max(board, y)
    x = min(height - board, x)
    y = min(width - board, y)
    img = img[x - board : x + board, y - board : y + board]
    img = cv2.resize(img, (i, j))
    return img

augments = ImageDataGenerator(
        featurewise_center=True,  # input mean to 0 over the all the data
        # samplewise_center=False,  # sample mean to 0
        featurewise_std_normalization=True,  # divide by std of the dataset
        # samplewise_std_normalization=False,  # divide each input by its std
        # zca_whitening=False,  # 
        rotation_range=10,  # random rotation, 0 to 180
        zoom_range = 0.10, # Randomly zoom in 
        width_shift_range=0.10,  # random horizontally, .01 to .99, fraction of width
        height_shift_range=0.10,  # random vertical shift. .01 to .99 fraction of height
        horizontal_flip=True,  # flip images horizontal, random
        rescale= 1.0/255,
        fill_mode='nearest',
        vertical_flip=True
        )  # flip images vertical, random 

import random
def prepareImages(data, m, dataset, i=150, j = 150, depth = 1):
    X = []
    Y = []
    count = 0
    for index, row in data.iterrows():
        fig = row['id']
        if (count%500 == 0):
            print("Loading image: ", count+1, ", ", fig)
        y = row['x']
        x = row['y']
        folder_name = fig[0:2]
        img_b = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_b.jpg")
        img_c = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_c.jpg")
        # img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        # img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

        if dataset == 'train':
            for tt in range(1):
                height, width, channel = img_b.shape
                back_x = random.randint(1,height)
                back_y = random.randint(1,width)
                while abs(back_x - x) < 15 or abs(back_y - y) < 15:
                    back_x = random.randint(1,height)
                    back_y = random.randint(1,width)

                img_back_b = crop_img(img_b, back_x, back_y, i, j)
                img_back_c = crop_img(img_c, back_x, back_y, i, j)
                X += [[img_back_b,img_back_c]]
                Y += [0]
                count += 1
        else:
            for tt in range(1):
                height, width, channel = img_b.shape
                back_x = random.randint(1,height)
                back_y = random.randint(1,width)
                while abs(back_x - x) < 15 or abs(back_y - y) < 15:
                    back_x = random.randint(1,height)
                    back_y = random.randint(1,width)

                img_back_b = crop_img(img_b, back_x, back_y, i, j)
                img_back_c = crop_img(img_c, back_x, back_y, i, j)
                X += [[img_back_b,img_back_c]]
                Y += [0]
                count += 1

        img_b = crop_img(img_b, x, y, i, j)
        img_c = crop_img(img_c, x, y, i, j)
        # print(img_b.shape)

        if dataset == 'train':
            for tt in range(1):
                # if is_new_star[row['judge']] == 1:
                #     for xx in range(4):
                #         img_tmp_b=img_b.reshape((1,) + img_b.shape)
                #         img_tmp_c=img_c.reshape((1,) + img_c.shape)
                #         augments.fit(img_tmp_b)
                #         augments.fit(img_tmp_c)
                #         img_tmp_b=img_tmp_b.reshape(i, j, depth)
                #         img_tmp_c=img_tmp_c.reshape(i, j, depth)
                #         X += [[img_tmp_b,img_tmp_c]]
                #         Y += [is_new_star[row['judge']]]
                img_b=img_b.reshape((1,) + img_b.shape)
                img_c=img_c.reshape((1,) + img_c.shape)
                augments.fit(img_b)
                augments.fit(img_c)
                img_b=img_b.reshape(i, j, depth)
                img_c=img_c.reshape(i, j, depth)
                X += [[img_b,img_c]]
                Y += [is_new_star[row['judge']]]
                # Y += [1]
                count += 1
        elif dataset == 'validation':
            X += [[img_b,img_c]]
            Y += [is_new_star[row['judge']]]
            count += 1
        
        # img = img.reshape(i, j, 1)
        # X_train[count] = img
        # if row['judge'] != 'known':
        # print(row['judge'])
        # print(x, y)
        # if row['judge'] != 'noise':
        # cv2.imshow("img", img_b)
        # q = cv2.waitKey(0)
        # cv2.imshow("img", img_c)
        # q = cv2.waitKey(0)
        # Y.append(int(is_new_star[row['judge']]))
        
    # X_train = X_train[0 : count, : , : , : ]
    return np.array(X), np.array(Y)

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
X_val, y_val = prepareImages(validation_df, validation_df.shape[0], "validation", height, width, depth)

steps_epoch = X.shape[0] // batch_size

base_network = cnn_model(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
from keras.optimizers import RMSprop
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer='adam', metrics=[accuracy])
# model.summary()

def scheduler(epoch):
    if epoch == 20:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

from keras.callbacks import *

reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(
        x = [X[:, 0],X[:, 1]],
        y = y,
        batch_size=batch_size,  
        callbacks = [#tensorboard, checkpointer ,
        reduce_lr,
        early_stopping,
        #wechat_utils.sendmessage(savelog=True,fexten='TEST')
        ],
        epochs = num_iter,
        validation_data = ([X_val[:,0],X_val[:,1]], y_val),
        verbose=1
        )

model.save_weights('models/siamese_cnn_model.h5')
