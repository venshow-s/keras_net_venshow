#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 20:49:19 2018

@author: szu
"""

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# global variable
batch_size=128
nb_class=10
epochs=12

img_rows,img_cols=28,28
nb_filters=32
pool_size=(2,2)
kernel_size=(3,3)

(x_train,y_train),(x_test,y_test)=mnist.load_data()


if K.image_dim_ordering()=='th':
    x_train=x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test=x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape=(1,img_rows,img_cols)
else:
    x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
    input_shape=(img_rows,img_cols,1)
#该选择结构可以灵活对theano和tensorflow两种backend生成对应格式的训练数据格式。
#举例说明：'th'模式，即Theano模式会把100张RGB三通道的16×32（高为16宽为32）
#彩色图表示为下面这种形式（100,3,16,32），Caffe采取的也是这种方式。第0个维度是样本维，
#代表样本的数目，第1个维度是通道维，代表颜色通道数。后面两个就是高和宽了。而TensorFlow，
#即'tf'模式的表达形式是（100,16,32,3），即把通道维放在了最后。这两个表达方法本质上没有什么区别。


x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /=255
x_test /=255

print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

#create model
model=Sequential()
model.add(Convolution2D(nb_filters,(kernel_size[0],kernel_size[1]),padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))#the first fully connective convolution
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_class)) #the sescond fully connective convolution
model.add(Activation('softmax')) 

# compile model
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#train model
#model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data(x_test,y_test))
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,  
          verbose=1, validation_data=(x_test, y_test))  
score=model.evaluate(x_test,y_test,verbose=0)
print('Test score:',score[0])
print('Test accuracy',score[1])
