import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print(f'Shape of x_train: {x_train.shape}')
print(f'Shape of y_train: {y_train.shape}')

# Normalize data
x_train = x_train.astype('float')
x_test = x_test.astype('float')
x_train /= 255
x_test /= 255
print(type(x_train))
plt.imshow(x_train[0])
plt.show()

input_shape = (32, 32, 3)

# Build model
model = Sequential([
    Conv2D(32, kernel_size = (3, 3), input_shape = input_shape),

    # Reduce size of input to increase computation time
    MaxPooling2D(pool_size = (3, 3)),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# Compule model 
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs = 15)

# Test model
score = model.evaluate(x_test, y_test)
print(score)
model.save('ml_cifar10_model.h5')