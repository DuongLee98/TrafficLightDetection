

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
import keras
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def rotation(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


print("Loading image...")
imagePaths = list(paths.list_images("dataset"))
random.seed(42)
random.shuffle(imagePaths)
clasf = 3
labels = []
data = []
eye = np.eye(clasf, dtype=int)
print(eye)

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (90, 90))
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(np.asarray(eye[int(label), :]))

data = np.array(data, dtype="float") / 255.0
print(data.shape)
labels = np.array(labels)
print(labels.shape)

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.2, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
print(trainX.shape)
testY = lb.transform(testY)

model = Sequential()

model.add(Conv2D(32, (7, 7), input_shape=(90, 90, 3), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Conv2D(64, (5, 5), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="SAME", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="SAME"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=clasf, activation='softmax'))

print(model.summary())

opt = SGD(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network
print("Training network...")
history = model.fit(trainX, trainY, epochs=60, validation_data=(testX, testY))

# Save the network to disk
print("Saving model....")
model.save("model.h5")
print("saved!")



