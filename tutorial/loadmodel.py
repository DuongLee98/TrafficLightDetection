import Model
import cv2
# import
model = Model.Model(modelSaveName="model.h5")
model.loadding()

# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

img = cv2.imread("../demo/img/IMG_1160.JPG")
# print(img)
print(model.predict([img]))