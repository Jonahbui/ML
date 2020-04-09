# Author: Jonah Bui
# Date: 3/10/2020
# Description: Used to allow the model to predict a set of images
# Best validation accuracy: 0.9941
# Changelog:
# 4/8/2020 - Removed unecessary packages, added more comments
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

#--------------------------------------------------------------------------------------------------
# Input
#--------------------------------------------------------------------------------------------------
data_gen = ImageDataGenerator(rescale = 1./255)

# Directory to pull images from to predict
prediction_dir = 'test'

# class_mode is none, since we want the model to predict the labels
test_gen_data = data_gen.flow_from_directory(prediction_dir,target_size=(128,128), class_mode=None,interpolation='bilinear')

labels = ['cat', 'dog']

model = keras.models.load_model('ml_cat_dog.h5')
#--------------------------------------------------------------------------------------------------
# Model Prediction
#--------------------------------------------------------------------------------------------------
prediction = model.predict(test_gen_data)

#--------------------------------------------------------------------------------------------------
# Results
#--------------------------------------------------------------------------------------------------
# Prints the prediciton of all the input images
for pred in prediction:
    label = round(pred[0]).astype(int)
    if label == 1:
        print("Dog")
    else:
        print("Cat")