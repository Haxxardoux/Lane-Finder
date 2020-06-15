from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from keras import backend as K
import os
import datetime

# Convolutional network experiencing memory issues when compiling, allow gpu memory growth to resolve this. 
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7 # try various numbers here
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# Probably temp
from PIL import Image
import matplotlib.pyplot as plt
import pickle



####  Load images one by one ####  
# image = np.asarray(Image.open('C:/Users/turbo/Desktop/download.jpg'))
# image_array = image.copy()
# weights = [0.2989, 0.5870, 0.1140]
# gray_img = np.dot(image_array[:,:,:3], weights)/255

# Data cleaning for mnist, if we want that
train = pd.read_csv('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/data/mnist_train.csv').to_numpy()
test = pd.read_csv('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/data/mnist_test.csv').to_numpy()

y_array = np.hstack((train[:,0], test[:,0]))
x_array = np.vstack((train[:,1:], test[:,1:])) / 255

y_test_array = keras.utils.to_categorical(y_array)

# Split test/train 
cv_pct = 0.2
idx = np.random.randint(len(x_array), size = round(len(x_array)*cv_pct))
mask = np.ones(x_array.shape[0], dtype = bool)
mask[idx] = False
x_train = x_array[mask]
x_test = x_array[idx]

# Reshape input because tensorflow is a big doo doo head and doesnt let us use binary images
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = y_test_array[mask]
y_test = y_test_array[idx]

swap_binary = True
if swap_binary: 
    #y_train -= 1
    x_train -= 1
    #y_train = abs(y_train)
    x_train = abs(x_train)

# Batch for training
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(100)
test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

#### Save image to desktop #####
#np.savetxt("foo2.csv", x_train[0].reshape(28,28), delimiter=",")

# Helper functions for min pooling 
def min_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    shape[2] /= 2
    shape[3] /= 2
    return tuple(shape)
def min_pool2d(x):
    return -K.pool2d(-x, pool_size=(2, 2), strides=(0,0))

# Models for min and max pooling
def max_pool_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=3,padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, kernel_size=3,padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, kernel_size=3,padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D(padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model


def min_pool_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=3,padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Lambda(min_pool2d, output_shape=min_pool2d_output_shape),
    tf.keras.layers.Conv2D(32, kernel_size=3,padding='same', activation='relu'),
    tf.keras.layers.Lambda(min_pool2d, output_shape=min_pool2d_output_shape),
    tf.keras.layers.Conv2D(64, kernel_size=3,padding='same', activation='relu'),
    tf.keras.layers.Lambda(min_pool2d, output_shape=min_pool2d_output_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    model.summary()

    model.compile(optimizer = 'sgd', loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model

# Initiate and run min pooling model
with tf.distribute.MirroredStrategy().scope():
    min_model = min_pool_model()

logdir = os.path.join(".\\logs\\fit\\", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

min_model.fit(train, epochs = 45, callbacks = [tensorboard_callback], verbose=1)
# min_model.fit(train, epochs=45)

# Initiate and run max pooling
with tf.distribute.MirroredStrategy().scope():
    max_model = max_pool_model()

# max_model.fit(train, epochs=45)



