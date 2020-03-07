#Best accuracy: 99.4%

# Supress tensorflow cuda warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filename = 'ml_mnist_model.h5'

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

# Load in dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# dataset dimensions
# x = index of image
# y = row of image @ x
# z = column of image @ x

# Data set has 60,000 images
# Image size is 28x28 pxl
print(f"Shape of x_train input: {x_train.shape}")
print(f"Shape of y_train input: {y_train.shape}")


# Reshape the input to a 4D array so that we can implement with Keras API
# Need 4D because:
# x = # of images
# y = row of pixels in image of x
# z = column of pixles in image of x
# t = the color channel in the image of x (in this case, 1 since it's greyscale)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# The shape of the image is 28x28 pxl with 1 channel
input_shape = (28, 28, 1)

# Make sure values are float so that decimal points are retained after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the RGB by dividing by max RGB value
x_train /= 255
x_test /= 255

# Create sequential model
model = Sequential([
    Conv2D(28, kernel_size = (3,3), input_shape = input_shape),
    
    # Used to reduce size of input images
    MaxPooling2D(pool_size = (2,2)),

    # Turn image into 1D vector so we can connect them to a dense layer
    Flatten(),
    Dense(128, activation = 'relu'),

    # Prevent overfitting the model, by making each node do 'work'
    Dropout(0.2),

    # Connect to 10 outputs
    Dense(10, activation = 'softmax')
])

# Compiles the model
model.compile(optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Train the model
model.fit(x_train, y_train, epochs = 10)

# Test the model
score = model.evaluate(x_test, y_test)

# Print model statistics
print(f'The accuracy of the model is: {score[0]}')

# Save the model
model.save(filename)