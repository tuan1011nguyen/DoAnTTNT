import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define classes
classes = ['With Helmet', 'Without Helmet']

# Load data
X_train = np.load(r'D:\DoAnTriTueNhanTao\X_train.npy')
y_train = np.load(r'D:\DoAnTriTueNhanTao\y_train.npy')

# Define a simple CNN model
def create_simple_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(len(classes), activation='sigmoid'))
    return model

# Compile the model
model = create_simple_cnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('helmet_detector_cnn.h5')
