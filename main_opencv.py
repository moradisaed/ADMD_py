import sys
import cv2 as cv
import numpy as np

def mat2gray(src):
    nrows,ncols = src.shape
    dst = np.zeros((nrows,ncols))
    cv.normalize(src, dst, 0, 1, cv.NORM_MINMAX)
    return dst


imgOrig = cv.imread(sys.argv[1], 0)
img1 = np.double(imgOrig)
# cv.imshow('ImageWindow1', mat2gray(img1))
#cv.waitKey()

lMin = 7
targetSize = (lMin, lMin)
muImg = cv.blur(img1, targetSize, cv.BORDER_DEFAULT) 
# cv.imshow('ImageWindow2', mat2gray(muImg))
# cv.waitKey()

maskKrlDilate = np.zeros((2*lMin+1, 2*lMin+1), np.uint8)
maskKrlDilate[0, 0] = 1
maskKrlDilate[2*lMin, 2*lMin] = 1
maskKrlDilate[lMin, 0] = 1
maskKrlDilate[2*lMin, 0] = 1
maskKrlDilate[0, lMin] = 1
maskKrlDilate[0, 2*lMin] = 1
maskKrlDilate[lMin, 2*lMin] = 1
maskKrlDilate[2*lMin, lMin] = 1
# maskKrlDilate[lMin, lMin] = 1


temp1 = cv.dilate(muImg, maskKrlDilate, iterations=1)

out1 = img1-temp1
T,temp_thresh = cv.threshold(out1, 0, 1, cv.THRESH_TOZERO)
out = cv.pow(temp_thresh,2)
cv.imshow('ImageWindow3', mat2gray(out))
cv.waitKey()
