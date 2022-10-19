"""
Flann based matching - https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
"""

import cv2 as cv
from cv2 import drawKeypoints
from matplotlib import pyplot as plt

import numpy as np

def main():
    # Initialize the images.
    img1 = cv.imread("src/ball1.jpg")
    img2 = cv.imread("src/ball2.jpg")
    img3 = cv.imread("src/ball3.jpg")

    # SiftFlann(img1, img2, img3)
    BriskFlann(img1, img2, img3)



def SiftFlann(img1, img2, img3):
    # get imagge keypoints and descriptors with Sift.
    siftKeypointsImg1, siftDescriptorsImg1 = GetSIFTKeypointsAndDescriptors(img1)
    siftKeypointsImg2, siftDescriptorsImg2 = GetSIFTKeypointsAndDescriptors(img2)
    siftKeypointsImg3, siftDescriptorsImg3 = GetSIFTKeypointsAndDescriptors(img3)

    FLANNmatcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    knn_matches1_2 = FLANNmatcher.knnMatch(siftDescriptorsImg1, siftDescriptorsImg2, 2)
    knn_matches1_3 = FLANNmatcher.knnMatch(siftDescriptorsImg1, siftDescriptorsImg3, 2)
    ratio_thresh = 0.7
    
    # Matching on images 1 and 2
    good_matches1_2 = []
    for m,n in knn_matches1_2:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_2.append(m)
    
    img_matches1_2 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, siftKeypointsImg1, img2, siftKeypointsImg2, good_matches1_2, img_matches1_2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/FlannSift1.jpg", img_matches1_2)

    # Matching on images 1 and 3
    good_matches1_3 = []
    for m,n in knn_matches1_3:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_3.append(m)

    img_matches1_3 = np.empty((max(img1.shape[0], img3.shape[0]), img1.shape[1]+img3.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, siftKeypointsImg1, img3, siftKeypointsImg3, good_matches1_3, img_matches1_3, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/FlannSift2.jpg", img_matches1_3)


def BriskFlann(img1, img2, img3):
    # get imagge keypoints and descriptors with Brisk.
    briskKeypointsImg1, briskDescriptorsImg1 = getBRISKKeypointsAndDesciptors(img1)
    briskKeypointsImg2, briskDescriptorsImg2 = getBRISKKeypointsAndDesciptors(img2)
    briskKeypointsImg3, briskDescriptorsImg3 = getBRISKKeypointsAndDesciptors(img3)

    # get images as floats cause screw brisk/flann I guess
    briskDescriptorsImg1 = np.float32(briskDescriptorsImg1)
    briskDescriptorsImg2 = np.float32(briskDescriptorsImg2)
    briskDescriptorsImg3 = np.float32(briskDescriptorsImg3)

    FLANNmatcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    knn_matches1_2 = FLANNmatcher.knnMatch(briskDescriptorsImg1, briskDescriptorsImg2, 2)
    knn_matches1_3 = FLANNmatcher.knnMatch(briskDescriptorsImg1, briskDescriptorsImg3, 2)

    ratio_thresh = 0.7

    #Matching on images 1 and 2.
    good_matches1_2 = []
    for m,n in knn_matches1_2:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_2.append(m)

    img_matches1_2 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, briskKeypointsImg1, img2, briskKeypointsImg2, good_matches1_2, img_matches1_2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/FlannBrisk1.jpg", img_matches1_2)

    #Matching on images 1 and 3.
    good_matches1_3 = []
    for m,n in knn_matches1_3:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_3.append(m)

    img_matches1_3 = np.empty((max(img1.shape[0], img3.shape[0]), img1.shape[1]+img3.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, briskKeypointsImg1, img3, briskKeypointsImg3, good_matches1_3, img_matches1_3, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/FlannBrisk2.jpg", img_matches1_3)



    # result1 = cv.drawKeypoints(img1, briskKeypointsImg1, None)
    # cv.imshow("Brisk", result1)
    # cv.waitKey()

    
    # # Matching on images 1 and 2
    # good_matches1_2 = []
    # for m,n in knn_matches1_2:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches1_2.append(m)
    # img_matches1_2 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, briskKeypointsImg1, img2, briskKeypointsImg2, good_matches1_2, img_matches1_2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # save matched images
    # cv.imwrite("src/FlannBrisk1.jpg", img_matches1_2)

    # # Matching on images 1 and 3
    # good_matches1_3 = []
    # for m,n in knn_matches1_3:
    #     if m.distance < ratio_thresh * n.distance:
    #         good_matches1_3.append(m)

    # img_matches1_3 = np.empty((max(img1.shape[0], img3.shape[0]), img1.shape[1]+img3.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, briskKeypointsImg1, img3, briskKeypointsImg3, good_matches1_3, img_matches1_3, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # save matched images
    # cv.imwrite("src/FlannBrisk2.jpg", img_matches1_3)



def GetSIFTKeypointsAndDescriptors(img):
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    return sift.detectAndCompute(imggray, None)

def getBRISKKeypointsAndDesciptors(img):
    # imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    BRISK = cv.BRISK_create()
    return BRISK.detectAndCompute(img, None)

def getFastKeypointsAndDesciptors(img):
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