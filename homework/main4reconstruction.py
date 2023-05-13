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

from submission import eightpoint, essentialMatrix, triangulate, epipolarCorrespondence
from helper import displayEpipolarF, camera2, epipolarMatchGUI
from findM2 import findM2
from visualize import visualize

if __name__ == "__main__":
    img1 = cv.imread('data/image1.jpg')
    img2 = cv.imread('data/image2.jpg')

    print(img1.shape)

    H, W, _ = img1.shape

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher()
    matches = flann.knnMatch(des1,des2,k=2)

    # orb = cv.ORB_create()
    # bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
    #matches = bf.knnMatch(des1, des2,k=2)

    src = []
    dst = []
    good = []

    # Ratio test
    for m, n in matches:
        if m.distance < 0.8*n.distance:
            src.append(kp1[m.queryIdx].pt)
            dst.append(kp2[m.trainIdx].pt)
            good.append([m])
    
    src = np.float32(src).reshape(-1, 2)
    dst = np.float32(dst).reshape(-1, 2)

    _, inliers = ransac((src, dst), AffineTransform, min_samples=4,
                        residual_threshold=8, max_trials=20000)
    
    src = src[inliers]
    dst = dst[inliers]

    # np.savez('q2.3_1.npz', src=src, dst=dst)

    img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, good,
                                    None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



    F = eightpoint(src, dst, max(H, W))

    # print("Fundamental Matrix: ")
    # print(F)
    # print('\n')
    # displayEpipolarF(img1, img2, F)

    # Load Intrinsic Matrix 
    K = {}
    with open('data/Intrinsic4Recon.npz', 'r') as f:
        for line in f:
            for value in line.split():
                if value == 'K1:' or value == 'K2:':
                    key = value[:2]
                    K[key] = []
                else:
                    K[key].append(float(value))

    K1 = np.array([K['K1']]).reshape((3,3))
    K2 = np.array([K['K2']]).reshape((3,3))
    visualize(img1, img2, F, K1, K2)

    # E = essentialMatrix(F, K1, K2)
    # M2s = camera2(E)
    
    # pts1, pts2 = epipolarMatchGUI(img1, img2, F)

    # np.savez('q2.5_1.npz', F=F, pts1=pts1, pts2=pts2)
    
    # print(pts1)
    # print(pts2)

    # print(type(M2s))
    # findM2(M2s, K1, K2)
    


