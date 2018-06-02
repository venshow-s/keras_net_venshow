#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 15:08:55 2018

@author: szu
"""

import csv
import cv2
import numpy as np

#creat model libary
from keras.models import Sequential
from keras.layers import Conv2D,Cropping2D
from keras.layers import Flatten,Dense,Lambda,Dropout

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#load train data
LINES=[]

CORRECTION=0.2

with open('./data/driving_log.csv') as csvfile:
    peader=csv.reader(csvfile)
    for line in peader:
        LINES.append(line)
        
train_samples,validation_samples=train_test_split(LINES,test_size=0.2)

#generat training samples 
def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1: #loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
        images=[]
        measurements=[]
        for batch_sample in batch_samples:
            measurement=float(batch_sample[3])
            measurement_left=measurement+CORRECTION
            measurement_right=measurement-CORRECTION
            
            for i in range(3):
                source_path=batch_sample[i]
                filename=source_path.split('/')[-1]
                current_path='./data/IMG/'+filename
                imge_bgr=cv2.imread(current_path)
                
                images.append(cv2.flip(image,1))
                
                measurements.extend([measurement,measurement*-1,
                                     measurement_left,measurement_left*-1,
                                     measurement_right,measurement_right*-1])
    
    x_train=np.array(images)
    y_train=np.array(measurements)
    
    yield shuffle(x_train,y_train)
    
    
train_generator=generator(train_samples,batch_size=32)
validation_generator=generator(validation_samples,batch_size=32)

row,col,ch=160,320,3

#model
model=Sequential()
model.add(Lambda(lambda x:x/127.5 -1.0,input_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((60,20),(0,0))))

model.add(Conv2D(24,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.7))
model.add(Conv2D(36,5,strides=(2,2),activation='relu'))
model.add(Conv2D(48,5,strides=(2,2),activation='relu'))
model.add(Conv2D(64,3,activation='relu'))
model.add(Conv2D(64,3,activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator,steps_per_epcoch=len(train_samples),
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples),
                    epochs=3)
model.save('')
