# Supress tensorflow cuda warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import z
# 
train_gen = ImageDataGenerator(rescale=1./255)
train_data_gen = train_gen.flow_from_directory('train',target_size=(150,150),batch_size=100,class_mode='binary',interpolation='bilinear')

val_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = val_gen.flow_from_directory('validation',target_size=(150,150),batch_size=100,class_mode='binary',interpolation='bilinear')

test_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = test_gen.flow_from_directory('validation',target_size=(150,150),batch_size=100,class_mode='binary',interpolation='bilinear')

input_shape = (150, 150, 3)

# Create sequential model
model = Sequential([
    Conv2D(32, kernel_size = (3,3), input_shape = input_shape),
    
    # Used to reduce size of input images
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(64, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(128, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),

    Conv2D(128, kernel_size = (3,3)),
    MaxPooling2D(pool_size = (2,2)),
    # Turn image into 1D vector so we can connect them to a dense layer
    Flatten(),
    Dense(150, activation = 'relu'),

    # Prevent overfitting the model, by making each node do 'work'
    Dropout(0.2),

    # Connect to 10 outputs
    Dense(1, activation = 'sigmoid')
])
model.summary()

# Compiles the model
model.compile(optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

# Train the model
model.fit_generator(train_data_gen, steps_per_epoch = 200,epochs = 10, validation_data = val_data_gen, validation_steps=25)

# Test the model
score = model.evaluate_generator(test_data_gen)

# Save the model
model.save('ml_cat_dog.h5')