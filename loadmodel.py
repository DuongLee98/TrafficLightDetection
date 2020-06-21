from numpy import loadtxt
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = load_model('model.h5')

image = cv2.imread("test/5a3ef95fa5191ad7fe6deec38103b8a64bd3ef24.jpg")
image = cv2.resize(image, (90, 90))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
pred = model.predict(image)
# pred = pred.argmax(axis=1)[0]
print("Predicted= ", pred)
