# Supress tensorflow cuda warning
import os
from os import listdir
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
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

target_size = (150, 150)

data = np.load('cat_dog_dataset.npz')
x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

print(f'{type(x_train)} | Shape is {x_train.shape}')
print(f'{type(y_train)} | Shape is {y_train.shape}')
print(f'{type(x_test)} | Shape is {x_test.shape}')
print(f'{type(y_test)} | Shape is {y_test.shape}')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 0 is cat, 1 is dog

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
    Dense(128, activation = 'relu'),

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
model.fit(x_train, y_train, validation_split = .25, epochs = 10, batch_size=240, use_multiprocessing=True, shuffle=True)

# Test the model
score = model.evaluate(x_test, y_test)

# Save the model
model.save('ml_cat_dog_2_model.h5')