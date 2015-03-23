import cv2
import numpy as np
from copy import copy
from mytools import *

history = []
history_maxlen = 10
min_pixels = 5
alpha = 0.01
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print 'No device'
    exit(0)

cv2.namedWindow('frame')
cv2.namedWindow('areas')
cv2.namedWindow('contours')

def minpixels(x):
    pass

cv2.createTrackbar('minpixels', 'areas', 5, 1000, minpixels)

ret = False
while not ret:
    ret, frame = cap.read()
frame = (cv2.resize(frame, (300, 300)) / 255.0).astype(np.float32)

subt = cv2.BackgroundSubtractorMOG(10, 3, 0.8, 3)

while(True):
    ret, frame = cap.read()

    if ret:
        frame = (cv2.resize(frame, (300, 300)) / 255.0).astype(np.float32)

        mask_bin = subt.apply((frame * 255).astype(np.uint8))

        min_pixels = cv2.getTrackbarPos('minpixels', 'areas')
        areas = areas_two_pass(mask_bin, min_pixels) * 4
        contours_m = contours_moore(areas)

        ct = np.zeros_like(frame)
        cv2.drawContours(ct, contours_m, -1, (0, 0 ,255), 1)



        cv2.imshow('frame', cv2.resize(frame, (300, 300)))
        cv2.imshow('areas', cv2.resize(areas, (300, 300)))
        cv2.imshow('contours', cv2.resize(ct, (300, 300)))


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
