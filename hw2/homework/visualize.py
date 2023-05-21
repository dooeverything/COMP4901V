'''
Q2.5.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import ransac
from skimage.transform import AffineTransform

from submission import essentialMatrix, epipolarCorrespondence, triangulate, eightpoint
from helper import camera2

def findM2(M2s: list, 
           K1: np.array, 
           K2: np.array,
           src: np.array,
           dst: np.array) -> None:

    m1 = np.zeros((3, 4))
    m1[0,0] = m1[1,1] = m1[2,2] = 1

    C1 = K1 @ m1
    C2s = np.array([K2 @ M2s[:,:,0], K2 @ M2s[:,:,1],  
                    K2 @ M2s[:,:,2], K2 @ M2s[:,:,3]])

    Ps = []
    num_pos = np.zeros((4,))
    for i in range(4):
        P, _ = triangulate(C1, src, C2s[i], dst)
        pos_z = (P[:,2]>0).sum()
        num_pos[i] = pos_z
        Ps.append(P)

    max_idx = num_pos.argmax()

    # print(Ps.shape)

    # print(f"n_neg1: {n_negs[0]}, n_neg2: {n_negs[1]}, n_neg3 {n_negs[2]}, n_neg4: {n_negs[3]}")
    # print(f"smallest negative at {min_idx}")
    # np.savez('q2.4_3.npz', M2=M2s[:,:,idx], C2=C2s[idx,:,:], P=Ps[idx])

    return M2s[:,:,max_idx], C1, C2s[max_idx,:,:], Ps[max_idx]


def visualize(img1, img2, F, K1, K2):
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

    M2, C1, C2, P = findM2(M2s, K1, K2, src, dst)

    p = P[abs(P[:,0] - np.mean(P[:,0])) < 2 * np.std(P[:,0])]
    # Y = P[abs(P[:,1] - np.mean(P[:,1])) < 2 * np.std(P[:,1])]
    # Z = P[abs(P[:,2] - np.mean(P[:,2])) < 2 * np.std(P[:,2])]
    np.savez('q2.5_2.npz', M2=M2, C1=C1, C2=C2, P=P)
    print(p.shape)
    ax.scatter(p[:,0], p[:,1], p[:,2])

    plt.show()

    return

if __name__ == "__main__":
    img1 = cv.imread('data/image1.jpg')
    img2 = cv.imread('data/image2.jpg')

    print(img1.shape)

    H, W, _ = img1.shape

    sift = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # find best matches from stereo images
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matches = flann.knnMatch(des1,des2,k=2)

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

    F = eightpoint(src, dst, max(H, W))

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
