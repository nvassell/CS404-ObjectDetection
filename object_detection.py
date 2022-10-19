"""
Flann based matching - https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
"""

import cv2 as cv
from cv2 import drawKeypoints
from matplotlib import pyplot as plt

import numpy as np

def main():
    SiftFlann()




    # GetSIFTKeypoints("ball1.jpg")
    # SIFT()
    # BRISK()
    # FAST()

def SiftFlann():
    img1 = cv.imread("src/ball1.jpg")
    img2 = cv.imread("src/ball2.jpg")
    img2 = cv.imread("src/ball3.jpg")
    # Image 1 and 2 Matching
    siftKeypointsImg1, siftDescriptorsImg1 = GetSIFTKeypointsAndDescriptors("ball1.jpg")
    siftKeypointsImg2, siftDescriptorsImg2 = GetSIFTKeypointsAndDescriptors("ball2.jpg")
    siftKeypointsImg3, siftDescriptorsImg3 = GetSIFTKeypointsAndDescriptors("ball3.jpg")
    FLANNmatcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = FLANNmatcher.knnMatch(siftDescriptorsImg1, siftDescriptorsImg2, 2)
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, siftKeypointsImg1, img2, siftKeypointsImg2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Show detected matches
    cv.imshow('Flann/Sift Matches (1,2)', img_matches)
    cv.waitKey()




def GetSIFTKeypointsAndDescriptors(filename):
    img = cv.imread(f'src/{filename}')
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    return sift.detectAndCompute(imggray, None)

def getBRISKKeypointsAndDesciptors(filename):
    img = cv.imread(f'src/{filename}')
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brisk = cv.BRISK_create()
    return brisk.detectAndCompute(imggray, None)

def getFastKeypointsAndDesciptors(filename):
    img = cv.imread(f'src/{filename}')
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    return fast.detectAndCompute(imggray, None)

# def BRISK():
#     img = cv.imread("src/ball1.jpg")
#     imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     brisk = cv.BRISK_create()
#     # Secont parameter is option mask to pass function
#     kp = brisk.detect(imggray, None)
#     img = drawKeypoints(img, kp, outImage=None)
#     cv.imshow("image", img)
#     cv.imwrite("src/ball1BRISK.jpg", img)
#     cv.waitKey();

# def FAST():
#     origimg = cv.imread("src/ball1.jpg")
#     img = cv.cvtColor(origimg, cv.COLOR_BGR2GRAY)
#     fast = cv.FastFeatureDetector_create()
#     # Secont parameter is option mask to pass function
#     kp = fast.detect(img, None)
#     img = drawKeypoints(origimg, kp, outImage=None)
#     cv.imshow("image", img)
#     cv.imwrite("src/ball1FAST.jpg", img)
#     cv.waitKey();

if __name__ == "__main__":
    main()