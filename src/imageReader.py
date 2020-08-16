# import os
# import cv2
#
# ds_factor = 0.6
#
# frame = cv2.imread('./static/rescources/0.jpg')
# frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor,
#                            interpolation=cv2.INTER_AREA)
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
#
# face_cascade = cv2.CascadeClassifier("C:/Users/Liuyc/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
# haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')
#
# face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
#
# print(cv2_base_dir)
# print(haar_model)
# print(face_cascade)

# from wand.image import Image
# import numpy as np
# import cv2
#
#
# with Image(filename='checks.png') as img:
#     print(img.size)
#     img.virtual_pixel = 'transparent'
#     img.distort('barrel', (0.2, 0.0, 0.0, 1.0))
#     img.save(filename='checks_barrel.png')
#     # convert to opencv/numpy array format
#     img_opencv = np.array(img)
#
# # display result with opencv
# cv2.imshow("BARREL", img_opencv)
# cv2.waitKey(0)

# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# # Define camera matrix K
# K = np.array([[673.9683892, 0., 343.68638231],
#               [0., 676.08466459, 245.31865398],
#               [0., 0., 1.]])
#
# # Define distortion coefficients d
# # d = np.array([5.44787247e-02, 1.23043244e-01, -4.52559581e-04, 5.47011732e-03, -6.83110234e-01])
#
# # Read an example image and acquire its size
# img = cv2.imread("checks.png")
# h, w = img.shape[:2]
#
# # Generate new camera matrix from parameters
# newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w,h), 0)
#
# # Generate look-up tables for remapping the camera image
# mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)
#
# # Remap the original image to a new image
# newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
#
# # Display old and new image
# fig, (oldimg_ax, newimg_ax) = plt.subplots(1, 2)
# oldimg_ax.imshow(img)
# oldimg_ax.set_title('Original image')
# newimg_ax.imshow(newimg)
# newimg_ax.set_title('Unwarped image')
# plt.show()

import cv2
import numpy as np

src = cv2.imread("static/rescources/1.jpg")
# width = src.shape[1]
# height = src.shape[0]
#
# distCoeff = np.zeros((4, 1), np.float64)
#
# # TODO: add your coefficients here!
# k1 = -1e-4;  # negative to remove barrel distortion
# k2 = 0;
# p1 = 0;
# p2 = 0;
#
# distCoeff[0, 0] = k1;
# distCoeff[1, 0] = k2;
# distCoeff[2, 0] = p1;
# distCoeff[3, 0] = p2;
#
# # assume unit matrix for camera
# cam = np.eye(3, dtype=np.float32)
#
# cam[0, 2] = width / 2.0  # define center x
# cam[1, 2] = height / 2.0  # define center y
# cam[0, 0] = 8.5  # define focal length x
# cam[1, 1] = 8.5  # define focal length y
#
# # here the undistortion will be computed
# dst = cv2.undistort(src, cam, distCoeff)
#
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# from scipy.interpolate import UnivariateSpline
#
# def spreadLookupTable(x, y):
#   spline = UnivariateSpline(x, y)
#   return spline(range(256))
#
# def warmImage(image):
#     increaseLookupTable = spreadLookupTable([0, 20, 40, 128, 256], [0, 50, 100, 160, 256])
#     decreaseLookupTable = spreadLookupTable([0, 50, 100, 256], [0, 50, 110, 256])
#
#     red_channel, green_channel, blue_channel = cv2.split(image)
#     red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
#     blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
#     return cv2.merge((red_channel, green_channel, blue_channel))
# #
# cv2.imwrite('haha.jpg', warmImage(src))
