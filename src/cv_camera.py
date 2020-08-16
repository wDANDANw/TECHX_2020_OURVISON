import cv2

''
import time
import os
import numpy as np
import random
from src import image_counter
from general_detection import getLabelAndBoxHuman, classes
import math

from wand.image import Image

from cv_kernels import KERNEL

currentIndex = 0
global_kernel = KERNEL.normal

# defining face detector
face_cascade = cv2.CascadeClassifier("C:/Users/Liuyc/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
smirk = cv2.imread('./static/rescources/FACE.jpg')
ds_factor = 0.6

audio_changed = False
audio_processed = False

class_ids = []
indices = []
boxes = []

random_time = 0

from scipy.interpolate import UnivariateSpline

def spreadLookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # releasing camera
        self.video.release()

    def applyKernel(self, frame, kernel):
        try:
            processed = cv2.filter2D(frame, -1, kernel)
        except:
            return frame

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

    def blurHuman(self, frame):
        class_ids, boxes, indices = getLabelAndBoxHuman(frame)

        new_frame = frame.copy()

        i = random.choice(indices)[0]

        box = boxes[i]

        x1 = round(box[0])
        y1 = round(box[1])
        x2 = round(box[0] + box[2])
        y2 = round(box[1] + box[3])

        rect = new_frame[y1:y2, x1:x2]

        processed = self.applyKernel(rect, KERNEL.boxBlur)
        new_frame[y1:y2, x1:x2] = processed

        return new_frame

    def makeFaces(self, frame):
        # frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
        #                    interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            face = cv2.resize(smirk, None, fx=w/smirk.shape[0], fy=h/smirk.shape[1],
                               interpolation=cv2.INTER_AREA)
            frame[y:y+h, x:x+w] = face

        return frame

    def motionBlur(self, frame):
        # # Import the image
        # with Image.from_array(frame) as image:
        #     # Clone the image in order to process
        #     with image.clone() as blur:
        #         # Invoke motion_blur function with radius 25, sigma 3 and angle 45
        #         blur.motion_blur(100, 20, 0)
        #         # Save the image
        #         cv_blur = np.array(blur)
        #
        #         return cv_blurx

        global image_counter
        image_counter += 1
        offset = int(math.sin(image_counter) * 20)

        new_frame = frame.copy()
        new_frame[50:430, 50:590, 0] = frame[50:430, 50+offset:590+offset, 0]
        new_frame[50:430, 50:590, 1] = frame[50:430, 40-offset:580-offset, 1]

        return new_frame

    def processAll(self, frame):

        global image_counter
        global audio_changed
        image_counter += 1

        # BENG DI
        # warmed = self.warmEffect(frame)
        # distored = self.distort(warmed)
        # processed = self.applyKernel(distored, global_kernel)
        # return self.warmEffect(processed)

        BLURHUMAN_IMPACT = 52

        if (image_counter < 150): ## 0-5 Normal Vision
            return frame
        elif (image_counter < 210 - BLURHUMAN_IMPACT): ## 5-7 Shadow
            return self.blurHuman(frame)
        elif (image_counter < 270 - BLURHUMAN_IMPACT): ## 7-9 Faces
            return self.makeFaces(frame)
        elif (image_counter < 330 - BLURHUMAN_IMPACT): ## 9-11 Global Blur
            return self.applyKernel(frame, KERNEL.boxBlur2)
        elif (image_counter < 420 - BLURHUMAN_IMPACT):  ## 11-14 Global Blur
            warmed = self.warmEffect(frame)
            distored = self.distort(warmed)
            processed = self.applyKernel(distored, global_kernel)
            return self.warmEffect(processed)
        elif (image_counter < 540 - BLURHUMAN_IMPACT): ## 14-18 Camera
            return self.motionBlur(frame)
        elif (image_counter < 560 - BLURHUMAN_IMPACT): ## BLACK

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = (image_counter - 540)
            frame[:, :, 2] = frame[:, :, 2] + value

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        elif (image_counter < 600 - BLURHUMAN_IMPACT):

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            value = (image_counter - 560)
            frame[:, :, 2] = frame[:, :, 2] - value

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        else:
            frame[:, :, :] = 0
            return frame

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()

        newFrame = self.processAll(frame)

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

    def get_audio_flag(self):
        global audio_changed
        return audio_changed






##### TEST CODE
# cap = cv2.VideoCapture(1)
#
# counter = 0
#
# class_ids = []
# indices = []
# boxes = []
#
# while(True):
#     ret, frame = cap.read(1)
#
#     # counter += 1
#     # if (counter >= 30):
#     class_ids, boxes, indices = getLabelAndBoxHuman(frame)
#     #     counter = 0
#     #
#     # print(class_ids)
#     #
#     # #render detection
#     # count = 0
#     # for i in indices:
#     #     i = i[0]
#     #     box = boxes[i]
#     #
#     #     class_id = class_ids[count]
#     #     label = str(classes[class_id])
#     #
#     #     cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])), (0, 0, 0), 2)
#     #     cv2.putText(frame, label, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#     #     count +=1
#
#     # newFrame = frame.copy()
#     # newFrame[:, :, ] = (3.063218 * dx - 1.393325 * dy - 0.475802 * dz)
#     # dg = (-0.969243 * dx + 1.875966 * dy + 0.041555 * dz)
#     # db = (0.067871 * dx - 0.228834 * dy + 1.069251 * dz)
#     # newFrame[:, :, 0] = 0.114 * frame[:, :, 0] + 0.588 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
#
#     cv2.imshow('frame', frame)
#     # save = 'C:/Users/' + getUser + "/Desktop/Trial/"
#     # path = os.path.join(save, "xdd.jpg")
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()