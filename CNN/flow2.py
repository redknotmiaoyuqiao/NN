# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

import PIL.Image as Image

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Convolution2D
from keras import backend as K
from keras.optimizers import SGD

path='c:/flower_photos/'

def get_data(path):
    files = os.listdir(path)

    rows = 224
    cols = 224
    dims = 3
    imgCount = 0
    category_num = 0

    for category in files:
        print("----" + category)
        images = os.listdir(path + "/" + category)
        category_num = category_num + 1
        for image in images:
            image_path = path + "/" + category + "/" + image
            imgCount = imgCount + 1

    print("Image Count : %d" % imgCount)
    print("Category Count : %d" % category_num)


    data_set_x = np.zeros((imgCount, rows, cols, dims))
    data_set_y = np.zeros((imgCount, category_num))

    i = 0
    j = 0
    for category in files:
        print("----" + category)
        images = os.listdir(path + "/" + category)
        j = j + 1
        for image in images:
            image_path = path + "/" + category + "/" + image
            image = Image.open(image_path)
            image = image.resize((rows,cols),Image.ANTIALIAS)
            image_arr = np.array(image)

            data_set_x[i] = image_arr
            data_set_y[i] = j
            i = i + 1


    data_set_x = data_set_x.astype('float32')
    data_set_x /= 255

    return data_set_x,data_set_y

data,label = get_data(path)

#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)

x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))#再来一次卷积 生成64*224*224
model.add(MaxPooling2D((2,2), strides=(2,2)))#pooling操作，相当于变成64*112*112

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))#128*56*56

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))#256*28*28

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))#512*14*14

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))  #到这里已经变成了512*7*7
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())#压平上述向量，变成一维25088
model.add(Dense(1024, activation='relu'))#全连接层有4096个神经核，参数个数就是4096*25088
model.add(Dropout(0.5))#0.5的概率抛弃一些连接
model.add(Dense(1024, activation='relu'))#再来一个全连接
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

#编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
#训练模型
batch_size = 20
epochs = 1000
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))
#评估模型
score = model.evaluate(x_val, y_val, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
