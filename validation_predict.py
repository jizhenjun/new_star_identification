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

input_shape = (width, height, depth)
num_classes = 2

is_new_star = {"newtarget": 1, "isstar": 1, "asteroid": 1, "isnova": 1, "known": 1, "noise": 0, "ghost": 0, "pity": 0,}

train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")

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

model.load_weights('models/siamese_cnn_model.h5')
# model.summary()

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

def predict(data, m, dataset, i=150, j = 150, depth = 1, model = model):
    X = []
    Y = []
    count = 0
    total = 0
    ans = 0
    for index, row in data.iterrows():
        fig = row['id']
        if (count%50 == 0):
            print("Loading image: ", count+1, ", ", fig)
        y = row['x']
        x = row['y']
        if is_new_star[row['judge']] == 0:
            continue
        total += is_new_star[row['judge']]
        folder_name = fig[0:2]
        img_b = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_b.jpg")
        img_c = cv2.imread("data/af2019-cv-training-20190312/" + folder_name + "/" + fig + "_c.jpg")

        border = 8
        height, width, channel = img_b.shape
        pred_y = np.zeros((height // border, width // border), dtype = float)
        for ii in range(height // border):
            for jj in range(width // border):
                center_x = ii * border + border // 2
                center_y = jj * border + border // 2
                img_tmp_b = crop_img(img_b, center_x, center_y, i, j)
                img_tmp_c = crop_img(img_c, center_x, center_y, i, j)
                img_tmp_b = img_tmp_b / 255.0
                img_tmp_c = img_tmp_c / 255.0
                tmp = []
                tmp += [[img_tmp_b, img_tmp_c]]
                tmp = np.array(tmp)
                pred_y[ii][jj] = model.predict([tmp[:,0],tmp[:,1]])

        max1_x, max1_y = np.unravel_index(np.argpartition(pred_y.flatten(), -1)[-1], pred_y.shape)
        max2_x, max2_y = np.unravel_index(np.argpartition(pred_y.flatten(), -2)[-2], pred_y.shape)
        max3_x, max3_y = np.unravel_index(np.argpartition(pred_y.flatten(), -3)[-3], pred_y.shape)
        max1_x = max1_x * 32 + 16
        max1_y = max1_y * 32 + 16
        max2_x = max2_x * 32 + 16
        max2_y = max2_y * 32 + 16
        max3_x = max3_x * 32 + 16
        max3_y = max3_y * 32 + 16
        if abs(max1_x - x) * abs(max1_x - x) + abs(max1_y - y) * abs(max1_y - y) < 16 * 16 or \
                abs(max2_x - x) * abs(max2_x - x) + abs(max2_y - y) * abs(max2_y - y) < 16 * 16 or \
                abs(max3_x - x) * abs(max3_x - x) + abs(max3_y - y) * abs(max3_y - y) < 16 * 16: 
                    ans += 1
        count += 1
    print(ans, total)
    return ans/total

score = predict(validation_df, validation_df.shape[0], "test", height, width, depth, model)
print(score)
