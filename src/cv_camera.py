import cv2
import getpass
from PIL import Image
from colorbilndProcessor import ColorBlindConverter, cb_types
import time
import os
import numpy as np

cap = cv2.VideoCapture(1)
flag = True

def processFrame(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    converter = ColorBlindConverter(pil_image)
    mode = 'Deuteranopia'
    if (mode in cb_types.keys()):
        converter.convert(mode)

    processed = converter.getImage()

    # use numpy to convert the pil_image into a numpy array
    numpy_image = np.array(processed)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

while(True):
    ret, frame = cap.read(1)

    newFrame = processFrame(frame)

    cv2.imshow('frame',newFrame)

    getUser = getpass.getuser()
    # save = 'C:/Users/' + getUser + "/Desktop/Trial/"
    # path = os.path.join(save, "xdd.jpg")

    path = 'C:/Users/Liuyc/Desktop/Trial/xdd.jpg'
    cv2.imwrite(path, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()