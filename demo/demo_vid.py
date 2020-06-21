import cv2
import imutils as imutils
import numpy as np
from hashlib import sha1

def process(im):

    threshold = 100
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    cv2.imshow("top", tophat)
    ret, thresh = cv2.threshold(tophat, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.imshow("dist", dist_transform)


    ret, markers = cv2.connectedComponents(np.uint8(dist_transform))
    # markers = markers + 1
    # markers[unknown == 255] = 0
    watershed = cv2.watershed(im, markers)
    # im[watershed == -1] = [255, 0, 0]
    # print(ret)
    mask = np.zeros(im.shape)
    if ret < 100:
        id = np.array([val for val in range(1, ret+1)])
        area = np.array([np.sum(markers == val) for val in range(1, ret+1)])
        select = np.array((area > 100) & (area < 1500))
        ft = np.array(id[select == True])
        area = np.array(area[select == True])
        # print(ft)
        # print(area)

        for sc in ft:
            aw = watershed == sc
            pos = np.argwhere(aw == True)
            # print(pos.shape)
            ps1, ps2 = bounding_box(pos)
            # print(ps1)
            # print(ps2)
            color = (0, 255, 0)
            w = ps2[0] - ps1[0]+1
            h = ps2[1] - ps1[1]+1
            if (w > 5 and h > 20 and h < 100) or (h > 5 and w > 20 and w < 100):

                crop = im[ps1[1]:ps1[1]+h, ps1[0]:ps1[0]+w]
                if not crop.flags['C_CONTIGUOUS']:
                    crop = np.ascontiguousarray(crop)
                    h = sha1(crop)

                    # print(h.hexdigest())
                    cv2.imwrite("data/"+h.hexdigest()+".jpg", crop)
                    cv2.rectangle(im, ps1, ps2, color)
            # mask[watershed == sc] = 255
            # print(b)
    return im

def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return (min(y_coordinates), min(x_coordinates)), (max(y_coordinates), max(x_coordinates))

cap = cv2.VideoCapture('video/VID_20200620_174445.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[650:1000, 400:900]
    frame = np.array(frame)
    # print(frame.shape)
    # print(frame.dtype)
    frame = np.uint8(process(frame))
    # print(frame.shape)
    # print(frame.dtype)

    cv2.imshow('frame', frame)
    # print(frame.shape)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()