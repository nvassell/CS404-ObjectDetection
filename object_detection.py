import cv2 as cv
from cv2 import drawKeypoints

import numpy as np

def main():
    SIFT()
    BRISK()
    FAST()

def SIFT():
    img = cv.imread("src/ball1.jpg")
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    # Secont parameter is option mask to pass function
    kp = sift.detect(imggray, None)
    img = drawKeypoints(img, kp, outImage=None)
    cv.imshow("image", img)
    # save the image, cause why not
    cv.imwrite("src/ball1SIFT.jpg", img)
    cv.waitKey();

def BRISK():
    img = cv.imread("src/ball1.jpg")
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brisk = cv.BRISK_create()
    # Secont parameter is option mask to pass function
    kp = brisk.detect(imggray, None)
    img = drawKeypoints(img, kp, outImage=None)
    cv.imshow("image", img)
    cv.imwrite("src/ball1BRISK.jpg", img)
    cv.waitKey();

def FAST():
    origimg = cv.imread("src/ball1.jpg")
    img = cv.cvtColor(origimg, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    # Secont parameter is option mask to pass function
    kp = fast.detect(img, None)
    img = drawKeypoints(origimg, kp, outImage=None)
    cv.imshow("image", img)
    cv.imwrite("src/ball1FAST.jpg", img)
    cv.waitKey();

if __name__ == "__main__":
    main()