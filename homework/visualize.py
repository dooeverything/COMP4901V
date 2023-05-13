'''
Q2.5.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import cv2 as cv
import numpy as np
from findM2 import findM2
from submission import epipolarCorrespondence, essentialMatrix
from helper import camera2
import matplotlib.pyplot as plt

def visualize(img1, img2, 
              F, K1, K2):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d', elev=50, azim=-40)
    print("Visualization")

    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)

    H, W, _ = img1.shape

    pts1_x = []
    pts1_y = []
    with open('data/VisPts.npz', 'r') as f:
        for line in f:
            pts1 = line.split()
            pts1_x.append(int(pts1[0]))
            pts1_y.append(int(pts1[1]))
            
    pts1 = np.stack( (pts1_x, pts1_y), axis=1)

    N, _ = pts1.shape
    # print(pts1.shape)

    # Get all the correspondence points
    src_x = []
    src_y = []

    dst_x = []
    dst_y = []
    for i in range(N):
        x1 = pts1[i,0]
        y1 = pts1[i,1]

        x2, y2 = epipolarCorrespondence(img1, img2, F, x1, y1)
        # if x2 > 0 and x2 < W and y2 > 0 and y2 < H:
        src_x.append(x1)
        src_y.append(y1)

        dst_x.append(x2)
        dst_y.append(y2)

    src = np.stack((src_x, src_y), axis=1)
    dst = np.stack((dst_x, dst_y), axis=1)
    # print(dst)

    _, C1, C2, P, err = findM2(M2s, K1, K2, src, dst)

    # print(P)
    # print(f"with error {err}")
    ax.scatter(P[:,0], P[:,1], P[:,2])

    plt.show()

    return
