# Supress tensorflow cuda warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(rescale = 1./255)

# class_mode is none, since we want the model to predict the labels
test_gen_data = data_gen.flow_from_directory('test',target_size=(128,128), class_mode=None,interpolation='bilinear')

labels = ['cat', 'dog']

model = keras.models.load_model('ml_cat_dog.h5')

prediction = model.predict(test_gen_data)


# Output predictions
for pred in prediction:
    label = round(pred[0]).astype(int)
    if label == 1:
        print("Dog")
    else:
        print("Cat")

for image in test_gen_data[0]:
    plt.imshow(image)
    plt.show()