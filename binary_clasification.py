import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import tensorflow as tf

import os
import random
from PIL import Image

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, BatchNormalization,
                                     Dropout, Activation)
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    zoom_range=0.4,
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Data Loaders
train_dataset = train_datagen.flow_from_directory(
    directory=r'/content/dataset/DATASET/TRAIN',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=128,
    subset='training'
)

valid_dataset = valid_datagen.flow_from_directory(
    directory=r'/content/dataset/DATASET/TRAIN',
    target_size=(224, 224),
    class_mode='binary',
    batch_size=128,
    subset='validation'
)

# Load base model
base_model = VGG16(input_shape=(224, 224, 3),
                   include_top=False,
                   weights='imagenet')

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Functional API model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(512, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss='binary_crossentropy',
    metrics=['accuracy'],
    optimizer=adam
)

# Print summary
model.summary()

# Callbacks
earlystopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=2, verbose=1)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5,
    callbacks=[earlystopping],
    verbose=1
)

# Save model
model.save('my_model.keras')
