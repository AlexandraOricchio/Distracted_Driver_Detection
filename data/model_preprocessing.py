import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from shutil import copyfile

import tensorflow as tf
from datetime import datetime as dt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout


random.seed(29)
if not os.path.exists('cleanData'):
    os.makedirs('cleanData')
if not os.path.exists('cleanData/train'):
    os.makedirs('cleanData/train')
if not os.path.exists('cleanData/validation'):
    os.makedirs('cleanData/validation')



subdirs = [subdir for subdir in os.listdir('rawData/imgs/train') if os.path.isdir(os.path.join('rawData/imgs/train',subdir))]



for subdir in subdirs:
    subdir_path = os.path.join('rawData/imgs/train',subdir)
    train_subdir = os.path.join('cleanData/train',subdir)
    validation_subdir = os.path.join('cleanData/validation',subdir)
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)
    if not os.path.exists(validation_subdir):
        os.makedirs(validation_subdir)
        
    train_counter=0
    validation_counter=0
    for filename in os.listdir(subdir_path):
        if filename.endswith('.jpg'):
            fileparts = filename.split('.')
            
            if random.uniform(0,1) <= 0.8:
                copyfile(os.path.join(subdir_path,filename), os.path.join(train_subdir, str(train_counter)+'.'+fileparts[1]))
                train_counter += 1
            else:
                copyfile(os.path.join(subdir_path,filename), os.path.join(validation_subdir, str(validation_counter)+'.'+fileparts[1]))
                validation_counter += 1
    print('Copied '+str(train_counter)+' iamges to cleanData/train/'+subdir)
    print('Copied '+str(validation_counter)+' iamges to cleanData/validation/'+subdir)




train_gen = ImageDataGenerator(rotation_range = 30,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              rescale = 1./255,
                              shear_range = 0.2,
                              zoom_range = 0.2)
valid_gen = ImageDataGenerator(rescale = 1./255)
train_generator = train_gen.flow_from_directory('cleanData/train',
                                               target_size=(480,640),
                                               batch_size=32,
                                               class_mode='categorical')
valid_generator = valid_gen.flow_from_directory('cleanData/validation',
                                               target_size=(480,640),
                                               batch_size=32,
                                               class_mode='categorical')




model = Sequential()
model.add(Conv2D(filters=32,
                kernel_size=(4,4),
                strides=(1,1),
                padding="same",
                input_shape=(480,640,3),
                data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4),
                      strides = 4))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64,
                kernel_size=(4,4),
                strides=(1,1),
                padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4),
                      strides = 4))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128,
                kernel_size=(4,4),
                strides=(1,1),
                padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4,4),
                      strides = 4))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])



start = dt.now()
model.fit_generator(train_generator,
                   steps_per_epoch = 18008//32,
                   epochs=15,
                   validation_data = valid_generator,
                   validation_steps = 4416//32)
print(dt.now()-start)

model.save("DDD_model.h5")