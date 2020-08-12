import cv2
import getpass
import time
import os
import numpy as np

cap = cv2.VideoCapture(0)
flag = True
newColorRange = [255, 255, 255] #B, G, R

def processFrame(frame, newRange):
    wMax, hMax = frame.shape[:2]

    processed = frame.copy()

    for w in range(wMax):
        for h in range(hMax):
            processed[w, h, 0] = int(frame[w, h, 0] * newRange[0] / 255) #B
            processed[w, h, 1] = int(frame[w, h, 0] * newRange[1] / 255) #G
            processed[w, h, 2] = int(frame[w, h, 0] * newRange[2] / 255) #R

    return processed

counter = 0
while(True):
    ret, frame = cap.read(1)

    newFrame = processFrame(frame, newColorRange)

    cv2.imshow('frame',newFrame)

    getUser = getpass.getuser()
    # save = 'C:/Users/' + getUser + "/Desktop/Trial/"
    # path = os.path.join(save, "xdd.jpg")

    path = 'C:/Users/Liuyc/Desktop/Trial/xdd.jpg'
    cv2.imwrite(path, frame)
    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()