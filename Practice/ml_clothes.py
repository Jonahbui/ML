import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_labels)

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size =3, activation="relu", input_shape=(28, 28, 1)),
    keras.layers.Conv2D(32, kernel_size = 3,activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation ="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, validation_data = (test_images, test_labels), epochs = 3)

prediction = model.predict(test_images[:4])