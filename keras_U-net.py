#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 21:39:33 2018

@author: szu
"""

from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Cropping2D
from keras.layers import merge
from keras.optimizers import Adam
import numpy as np

seed=7
np.random.seed(seed)

inpt=Input(shape=(256,256,3))

conv1=Conv2D(64,3,activation='relu',padding='valid',kernel_initializer='uniform')(inpt)
conv1=Conv2D(64,3,activation='relu',padding='valid',kernel_initializer='uniform')(conv1)
crop1=Cropping2D(cropping=((90,90),(90,90)))(conv1)
pool1=MaxPooling2D(pool_size=(2,2))(conv1)

conv2=Conv2D(128,3,activation='relu',padding='valid',kernel_initializer='uniform')(pool1)
conv2=Conv2D(128,3,activation='relu',padding='valid',kernel_initializer='uniform')(conv2)
crop2=Cropping2D(cropping=((41,41),(41,41)))(conv2)
pool2=MaxPooling2D(pool_size=(2,2))(conv2)

conv3=Conv2D(256,3,activation='relu',padding='valid',kernel_initializer='uniform')(pool2)
conv3=Conv2D(256,3,activation='relu',padding='valid',kernel_initializer='uniform')(conv3)
crop3=Cropping2D(cropping=((16,17),(16,17)))(conv3)
pool3=MaxPooling2D(pool_size=(2,2))(conv3)

conv4=Conv2D(512,3,activation='relu',padding='valid',kernel_initializer='uniform')(pool3)
conv4=Conv2D(512,3,activation='relu',padding='valid',kernel_initializer='uniform')(conv4)
drop4=Dropout(0.5)(conv4)
crop4=(Cropping2D(cropping=((4,4),(4,4))))(drop4)
pool4=MaxPooling2D(pool_size=(2,2))(drop4)

conv5=Conv2D(1024,3,activation='relu',padding='valid',kernel_initializer='uniform')(pool4)
conv5=Conv2D(1024,3,activation='relu',padding='valid',kernel_initializer='uniform')(conv5)
drop5=Dropout(0.5)(conv5)

# upsample
up6=Conv2D(512,2,activation='relu',padding='same',kernel_initializer='uniform')(UpSampling2D(size=(2,2))(drop5))
merge6=merge([crop4,up6],mode='concat',concat_axis=3)
conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='uniform')(merge6)
conv6=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='uniform')(conv6)

up7=Conv2D(256,2,activation='relu',padding='same',kernel_initializer='uniform')(UpSampling2D(size=(2,2))(conv6))
merge7=merge([crop3,up7],mode='concat',concat_axis=3)
conv7=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='uniform')(merge7)
conv7=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='uniform')(conv7)

up8=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='uniform')(UpSampling2D(size=(2,2))(conv7))
merge8=(merge([crop2,up8],mode='concat',concat_axis=3))
conv8=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='uniform')(merge8)
conv8=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='uniform')(conv8)

up9=Conv2D(64,2,activation='relu',padding='same',kernel_initializer='uniform')(UpSampling2D(size=(2,2))(conv8))
merge9=merge([crop1,up9],model='concat',concat_axis=3)
conv9=Conv2D(64,3,activation='relu',padding='same',kerenel_initializer='uniform')(merge9)
conv9=Conv2D(2,3,activation='relu',padding='same',kernel_initializer='uniform')(conv9)

conv10=Conv2D(1,1,activation='sigmoid')(conv9)

model=Model(input=inpt,output=conv10)

model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
