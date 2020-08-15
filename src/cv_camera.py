import cv2
import getpass
from PIL import Image
from colorbilndProcessor import ColorBlindConverter, cb_types
from histogram_trial import gpu_enhanceImage
import time
import os
import numpy as np

cap = cv2.VideoCapture(1)
flag = True

def processFrame(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pil_image1 = pil_image.crop((80, 0, 560, 480))

    # data = np.zeros((480, 640, 3), dtype=np.uint8)
    # data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    # pil_image = Image.fromarray(data)

    converter = ColorBlindConverter(pil_image1)
    mode = 'Deuteranomaly'
    if (mode in cb_types.keys()):
        converter.convert(mode)

    processed = converter.getImage()

    # processed = gpu_enhanceImage(pil_image)

    # use numpy to convert the pil_image into a numpy array
    numpy_image = np.array(processed)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    return opencv_image

while(True):
    ret, frame = cap.read(1)

    newFrame = processFrame(frame)

    # newFrame = frame.copy()
    # newFrame[:, :, ] = (3.063218 * dx - 1.393325 * dy - 0.475802 * dz)
    # dg = (-0.969243 * dx + 1.875966 * dy + 0.041555 * dz)
    # db = (0.067871 * dx - 0.228834 * dy + 1.069251 * dz)
    # newFrame[:, :, 0] = 0.114 * frame[:, :, 0] + 0.588 * frame[:, :, 1] + 0.299 * frame[:, :, 2]

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