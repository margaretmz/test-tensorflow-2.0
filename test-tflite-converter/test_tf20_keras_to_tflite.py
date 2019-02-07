#
# Margaret Maynard-Reid
# 1/25/2019
# Testing TensorFlow 2.0 preview
#
# A simple CNN for MNIST, test convert Keras model to tflite in TF 2.0


"""### Installation & Imports"""

# Import TensorFlow and tf.keras
import tensorflow as tf
keras = tf.keras

# Import helper libraries
import numpy as np
import matplotlib.pyplot as plt

# Import date for tflite file name
from datetime import date

# Print TensorFlow version
version = tf.__version__
print("TensorFlow version: " + version)

"""### Get MNIST dataset"""

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

"""### Preprocess data"""

num_classes = 10
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize the input data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Take a look at the dataset shape after conversion with keras.utils.to_categorical
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

"""### Define the model architecture"""

model = keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

"""### Compile the model"""

model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.Adam(),
             metrics=['accuracy'])

"""### Train the model"""

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=1,
         validation_data=(x_test, y_test))

"""### Save a Keras model"""

# Save tf.keras model in HDF5 format
keras_model_file = "mnist_keras_model.h5"
keras.models.save_model(model, keras_model_file)

"""### Convert Keras model to TensorFlow Lite model"""
# Note: conversion to tflite works in tf 1.11.0 and 1.12.0, but not 2.0 preview
# converter=tf.contrib.lite.TocoConverter.from_keras_model_file(keras_model_file)   # TF 1.11.0
converter=tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model_file)   # TF 1.12.0
# converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_model_file)       # TF 2.0
tflite_model = converter.convert()
tflite_model_file_name = "mnist_" + version + "_" + str(date.today()) + ".tflite"
open(tflite_model_file_name, "wb").write(tflite_model)