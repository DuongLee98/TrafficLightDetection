import ObjectDetection as od
import Model
import cv2
import numpy as np
import time

#
model = Model.Model(modelSaveName="../model.h5")
model.loadding()

objd = od.ObjectMiniDetection()

cap = cv2.VideoCapture('../demo/video/VID_20200620_174445.mp4')
print(cap.isOpened())
totalFrame = 0
start = time.time()

while cap.isOpened():
    totalFrame += 1
    ret, frame = cap.read()
    frame = frame[650:1000, 400:900]
    frame = np.array(frame)

    _, _, _, objs, bound = objd.process(frame, max_object=10)
    if len(objs) > 0:
        predict = model.predict(objs)
        # print("Predict: ", predict)
    color = (0, 255, 0)
    for b in bound:
        cv2.rectangle(frame, b[0], b[1], color)

    end = time.time()
    fps = int(totalFrame / (end - start))
    print("Estimated frames per second : {0}".format(fps))

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
