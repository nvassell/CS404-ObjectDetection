"""
Flann based matching - https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
BF base matching - https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
More feature matching - https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
https://stackoverflow.com/questions/39940766/bfmatcher-match-in-opencv-throwing-error
"""

import cv2 as cv
from matplotlib import pyplot as plt

import numpy as np

def main():
    # Initialize the images.
    img1 = cv.imread("src/ball1.jpg")
    img2 = cv.imread("src/ball2.jpg")
    img3 = cv.imread("src/ball3.jpg")

    # SiftFlann(img1, img2, img3)
    # BriskFlann(img1, img2, img3)
    # OrbFlann(img1, img2, img3)
    # SiftBF(img1, img2, img3)
    # OrbBF(img1, img2, img3)
    BriskBF(img1, img2, img3)


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
    # get image keypoints and descriptors with Brisk.
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


def OrbFlann(img1, img2, img3):
    orbKeypoints1, orbDescriptorsImg1 = getOrbKeypointsAndDescriptors(img1)
    orbKeypoints2, orbDescriptorsImg2 = getOrbKeypointsAndDescriptors(img2)
    orbKeypoints3, orbDescriptorsImg3 = getOrbKeypointsAndDescriptors(img3)

    orbDescriptorsImg1 = np.float32(orbDescriptorsImg1)
    orbDescriptorsImg2 = np.float32(orbDescriptorsImg2)
    orbDescriptorsImg3 = np.float32(orbDescriptorsImg3)

    FLANNmatcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

    knn_matches1_2 = FLANNmatcher.knnMatch(orbDescriptorsImg1, orbDescriptorsImg2, 2)
    knn_matches1_3 = FLANNmatcher.knnMatch(orbDescriptorsImg1, orbDescriptorsImg3, 2)

    ratio_thresh = 0.7

    #Matching on images 1 and 2.
    good_matches1_2 = []
    for m,n in knn_matches1_2:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_2.append(m)

    img_matches1_2 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, orbKeypoints1, img2, orbKeypoints2, good_matches1_2, img_matches1_2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/OrbFlann1.jpg", img_matches1_2)

    #Matching on images 1 and 3.
    good_matches1_3 = []
    for m,n in knn_matches1_3:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_3.append(m)

    img_matches1_3 = np.empty((max(img1.shape[0], img3.shape[0]), img1.shape[1]+img3.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img1, orbKeypoints1, img3, orbKeypoints3, good_matches1_3, img_matches1_3, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/OrbFlann2.jpg", img_matches1_3)

def SiftBF(img1, img2, img3):
    # get imagge keypoints and descriptors with Sift.
    siftKeypointsImg1, siftDescriptorsImg1 = GetSIFTKeypointsAndDescriptors(img1)
    siftKeypointsImg2, siftDescriptorsImg2 = GetSIFTKeypointsAndDescriptors(img2)
    siftKeypointsImg3, siftDescriptorsImg3 = GetSIFTKeypointsAndDescriptors(img3)

    bf = cv.BFMatcher(cv.NORM_L1,crossCheck=False)
    matches1_2 = bf.match(siftDescriptorsImg1, siftDescriptorsImg2)
    matches1_3 = bf.match(siftDescriptorsImg1, siftDescriptorsImg3)

    # Sort them in the order of their distance.
    matches1_2 = sorted(matches1_2, key = lambda x:x.distance)
    matches1_3 = sorted(matches1_3, key = lambda x:x.distance)

    # Draw first 10 matches.
    image1_2 = cv.drawMatches(img1,siftKeypointsImg1,img2,siftKeypointsImg2,matches1_2[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image1_3 = cv.drawMatches(img1,siftKeypointsImg1,img3,siftKeypointsImg3,matches1_3[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the images :)
    cv.imwrite("src/BFSift1.jpg", image1_2)
    cv.imwrite("src/BFSift2.jpg", image1_3)

def BriskBF(img1, img2, img3):
    orbKeypoints1, orbDescriptorsImg1 = getOrbKeypointsAndDescriptors(img1)
    orbKeypoints2, orbDescriptorsImg2 = getOrbKeypointsAndDescriptors(img2)
    orbKeypoints3, orbDescriptorsImg3 = getOrbKeypointsAndDescriptors(img3)

    # create BFMatcher object
    bf = cv.BFMatcher()

    # Match descriptors.]
    matches1_2 = bf.knnMatch(orbDescriptorsImg1, orbDescriptorsImg2, k=2)
    matches1_3 = bf.knnMatch(orbDescriptorsImg1, orbDescriptorsImg3, k=2)

    ratio_thresh = .9

    #Matching on images 1 and 2.
    good_matches1_2 = []
    for m,n in matches1_2:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_2.append(m)

    #Matching on images 1 and 3.
    good_matches1_3 = []
    for m,n in matches1_3:
        if m.distance < ratio_thresh * n.distance:
            good_matches1_3.append(m)

    img_matches1_2 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    img_matches1_3 = np.empty((max(img1.shape[0], img3.shape[0]), img1.shape[1]+img3.shape[1], 3), dtype=np.uint8)

    cv.drawMatches(img1, orbKeypoints1, img2, orbKeypoints2, good_matches1_2, img_matches1_2, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.drawMatches(img1, orbKeypoints1, img3, orbKeypoints3, good_matches1_3, img_matches1_3, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # save matched images
    cv.imwrite("src/BriskBF1.jpg", img_matches1_2)
    cv.imwrite("src/BriskBF2.jpg", img_matches1_3)


def OrbBF(img1, img2, img3):
    orbKeypoints1, orbDescriptorsImg1 = getOrbKeypointsAndDescriptors(img1)
    orbKeypoints2, orbDescriptorsImg2 = getOrbKeypointsAndDescriptors(img2)
    orbKeypoints3, orbDescriptorsImg3 = getOrbKeypointsAndDescriptors(img3)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.]
    matches1_2 = bf.match(orbDescriptorsImg1, orbDescriptorsImg2)
    matches1_3 = bf.match(orbDescriptorsImg1, orbDescriptorsImg3)

    # Sort them in the order of their distance.
    matches1_2 = sorted(matches1_2, key = lambda x:x.distance)
    matches1_3 = sorted(matches1_3, key = lambda x:x.distance)

    # Draw first 10 matches.
    image1_2 = cv.drawMatches(img1,orbKeypoints1,img2,orbKeypoints2,matches1_2[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image1_3 = cv.drawMatches(img1,orbKeypoints1,img3,orbKeypoints3,matches1_3[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    # Draw the images
    cv.imwrite("src/BFOrb1.jpg", image1_2)
    cv.imwrite("src/BFOrb2.jpg", image1_2)



def GetSIFTKeypointsAndDescriptors(img):
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    return sift.detectAndCompute(imggray, None)

def getBRISKKeypointsAndDesciptors(img):
    imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    BRISK = cv.BRISK_create()
    return BRISK.detectAndCompute(imggray, None)

def getOrbKeypointsAndDescriptors(img):
    orb = cv.ORB_create()
    return orb.detectAndCompute(img, None)

if __name__ == "__main__":
    main()
