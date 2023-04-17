'''
It is a main function that can be used for you to conduct the tasks in 3D reconstruction.
You run this main function to generate the expected outputs and results described in the Instruction.pdf, 
by calling functions implemented in submission.py and helper.py
You are free to write it in your own style.
'''

# Insert your package here
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from skimage.measure import ransac
from skimage.transform import AffineTransform


if __name__ == "__main__":
    img_left = cv.imread('data/image1.jpg',0)
    img_right = cv.imread('data/image2.jpg',0)

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_left,None)
    kp2, des2 = sift.detectAndCompute(img_right,None)

    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv.FlannBasedMatcher()

    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    matches = bf.knnMatch(des1, des2,k=2)

    # matches = sorted(matches, key = lambda x:x.distance)

    src = []
    dst = []
    good = []

    # Ratio test
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            src.append(kp1[m.queryIdx].pt)
            dst.append(kp2[m.trainIdx].pt)
            good.append([m])
    
    src = np.float32(src).reshape(-1, 2)
    dst = np.float32(dst).reshape(-1, 2)

    _, inliers = ransac((src, dst), AffineTransform, min_samples=4,
                        residual_threshold=8, max_trials=10000)
    
    src = src[inliers]
    dst = dst[inliers]

    img_matches = cv.drawMatchesKnn(img_left, kp1, img_right, kp2, good,
                                 None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    

    plt.imshow(img_matches)
    plt.show()