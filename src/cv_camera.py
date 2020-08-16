import cv2

''
import time
import os
import numpy as np
import random
from src import image_counter

from cv_kernels import KERNEL

currentIndex = 0
global_kernel = KERNEL.normal

# defining face detector
face_cascade = cv2.CascadeClassifier("C:/Users/Liuyc/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
ds_factor = 0.6

from scipy.interpolate import UnivariateSpline

def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        # releasing camera
        self.video.release()

    def applyKernel(self, frame, kernel):
        processed = cv2.filter2D(frame, -1, kernel)
        return processed

    def distort(self, frame):
        cam = np.array([
            [7, 0., 320.],
            [0., 7, 240.],
            [0., 0., 1.]
        ])

        distCoeff = np.array([
            [-0.00027],
            [0.],
            [0.],
            [0.]
        ])

        return cv2.undistort(frame, cam, distCoeff)

    def warmEffect(self, frame):

        red_rand = random.randrange(20, 60)
        blue_rand = random.randrange(30, 40)

        increaseLookupTable = spreadLookupTable([0, red_rand, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = spreadLookupTable([0, blue_rand, 100, 256], [0, 50, 130, 256])

        red_channel, green_channel, blue_channel = cv2.split(frame)
        red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
        return cv2.merge((red_channel, green_channel, blue_channel))

    def manicEffect(self, frame):
        return

    def processAll(self, frame):

        global image_counter
        image_counter += 1

        if (image_counter >= 480):
            image_counter = 0
            return frame
        elif image_counter >= 240:
            warmed = self.warmEffect(frame)
            distored = self.distort(warmed)
            processed = self.applyKernel(distored, global_kernel)

            return self.warmEffect(processed)
        else:
            return frame

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()

        newFrame = self.processAll(frame)
        print(np.shape(newFrame))

        newFrame = newFrame[60:420, 80:560]
        # cv2.imshow('frame', newFrame)

        # frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
        #                    interpolation=cv2.INTER_AREA)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        # for (x, y, w, h) in face_rects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     break
        # encode OpenCV raw frame to jpg and displaying it

        # path = './static/rescources/' + str(1) + '.jpg'
        # cv2.imwrite(path, newFrame)

        ret, jpeg = cv2.imencode('.jpg', newFrame)
        return jpeg.tobytes()

# while(True):
#     ret, frame = cap.read(1)
#
#
#
#     # newFrame = frame.copy()
#     # newFrame[:, :, ] = (3.063218 * dx - 1.393325 * dy - 0.475802 * dz)
#     # dg = (-0.969243 * dx + 1.875966 * dy + 0.041555 * dz)
#     # db = (0.067871 * dx - 0.228834 * dy + 1.069251 * dz)
#     # newFrame[:, :, 0] = 0.114 * frame[:, :, 0] + 0.588 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
#
#
#     # save = 'C:/Users/' + getUser + "/Desktop/Trial/"
#     # path = os.path.join(save, "xdd.jpg")
#
#     # path = 'C:/Users/Liuyc/Desktop/Trial/xdd.jpg'
#
#     path = './static/rescources/' + str(image_counter) + '.jpg'
#     image_counter += 1
#     if (image_counter > 30):
#         path2 = './static/rescources/' + str(image_counter-30) + '.jpg'
#         os.remove(path2)
#
#     cv2.imwrite(path, newFrame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
