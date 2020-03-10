# Author: Jonah Bui
# Date: 3/10/2020
# Best validation accuracy: 0.9940
# Supress tensorflow cuda warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Enable GPU support using plaidml
from plaidml import keras
os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from PIL import Image
#import tensorflow as tf
#from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# Directory containing training data (subdirectories: cat, dog)
TRAIN_DIR = 'train'

# Get images from directory to feed into model and label them based of subdirectory name
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data_gen = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128,128),
    batch_size=128,
    class_mode='binary',
    interpolation='bilinear')

val_data_gen = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128,128),
    batch_size=128,
    class_mode='binary',
    interpolation='bilinear'
)

# Shape of input images
input_shape = (128, 128, 3)

# Create sequential model
model = Sequential([
    Conv2D(32, kernel_size = (3,3), input_shape = input_shape, padding='same'),
    
    # Used to reduce size of input images
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(64, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(128, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(64, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),

    # Turn image into 1D vector so we can connect them to a dense layer
    Flatten(),
    Dense(1024, activation = 'relu'),

    # Prevent overfitting the model, by making each node do 'work'
    Dropout(0.3),

    # Fire if dog, not if cat
    # Sigmoid since binary classification
    Dense(1, activation = 'sigmoid')
])
model.summary()

# Compiles the model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Train the model
model.fit_generator(
    train_data_gen,
    epochs = 15, 
    validation_data = val_data_gen, 
    steps_per_epoch=196, 
    validation_steps=196
)

# Save the model
model.save('ml_cat_dog.h5')