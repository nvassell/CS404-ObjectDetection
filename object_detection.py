import cv2 as cv
import numpy as np



def SIFT():
    img = cv.imread("src/ball1.jpg")
    key = ord('r')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    # Secont parameter is option mask to pass function
    kp = sift.detect(img, None)
    img = drawKeypoints(img, kp, outImage=None)
    cv.imshow("image", img)
    while key != ord('s'):
        key = cv.waitKey(5);

def BRISK():
    img = cv.imread("src/ball1.jpg")
    key = ord('r')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    brisk = cv.BRISK_create()
    # Secont parameter is option mask to pass function
    kp = brisk.detect(img, None)
    img = drawKeypoints(img, kp, outImage=None)
    cv.imshow("image", img)
    while key != ord('s'):
        key = cv.waitKey(5);

def FAST():
    key = ord('r')
    origimg = cv.imread("src/ball1.jpg")
    img = cv.cvtColor(origimg, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    # Secont parameter is option mask to pass function
    kp = fast.detect(img, None)
    img = drawKeypoints(origimg, kp, outImage=None)
    cv.imshow("image", img)
    while key != ord('s'):
        key = cv.waitKey(5);

def blob():
    key = ord('r')
    origimg = cv.imread("src/ball1.jpg")
    img = cv.cvtColor(origimg, cv.COLOR_BGR2GRAY)
    blob = cv.SimpleBlobDetector_create()
    # Secont parameter is option mask to pass function
    kp = blob.detect(img, None)
    img = drawKeypoints(origimg, kp, outImage=None, color=(0,0,225), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("image", img)
    while key != ord('s'):
        key = cv.waitKey(5);\

def Harris():
    img = cv.imread("src/ball1.jpg")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    dst = cv.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)

    # dilate to mark the corners
    dst = cv.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 255, 0]

    cv.imshow('haris_corner', img)
    cv.waitKey()

def Orb():
    img1 = cv.imread("src/ball3.jpg", 0)
    img2 = cv.imread("src/ball2.jpg", 0)

    orb = cv.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 50 matches
    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv.imshow('Matches', match_img)
    cv.waitKey()

# SIFT()
# BRISK()
# FAST()
# blob()
# Harris()
Orb()