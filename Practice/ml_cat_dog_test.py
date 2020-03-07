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

#test_gen = ImageDataGenerator(rescale = 1./255)
#test_gen_data = test_gen.flow_from_directory('eval',target_size=(150,150), class_mode='binary',interpolation='bilinear', save_to_dir='new')

labels = ['cat', 'dog']

model = keras.models.load_model('ml_cat_dog.h5')

img = pltimg.imread('new/_9_1118471.png')
plt.imshow(img)
plt.show()
img = img.reshape(1, 150, 150, 3)
print(f'Shape of img: {img.shape}')

prediction = model.predict(img)
print(prediction)