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

from tensorflow.keras.models import load_model


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

model = load_model("48_model.h5")

model.evaluate(train_generator)
model.evaluate(valid_generator)