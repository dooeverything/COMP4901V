"""
Homework2.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import os

from helper import *
import cv2 as cv
import numpy as np

from skimage.measure import ransac
from skimage.transform import AffineTransform
from tqdm import tqdm

'''
Q2.3.2: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M) -> np.array:
        
    N, _ = pts1.shape
    T = np.array([ [1.0/M,    0,   0],
                   [0,    1.0/M,   0],
                   [0,        0,   1] ])
    # m1 = pts1.mean(axis=0)
    # m2 = pts2.mean(axis=0)
    
    # m1 = np.array([ [1, 0, -m1[0]],
    #                 [0, 1, -m1[1]],
    #                 [0, 0,      0] ])

    # m2 = np.array([ [1, 0, -m2[0]],
    #                 [0, 1, -m2[1]],
    #                 [0, 0,      0] ])
    # T1 = T @ m1
    # T2 = T @ m2

    ones = np.ones((N,1))  
    pts1_T = np.append(pts1, ones, axis=1).T
    pts2_T = np.append(pts2, ones, axis=1).T
    pts1_norm = (T @ pts1_T).T
    pts2_norm = (T @ pts2_T).T
    
    A1 = (pts1_norm[:,0] * pts2_norm[:,0]).reshape((N,1))
    A2 = (pts1_norm[:,0] * pts2_norm[:,1]).reshape((N,1))
    A3 = pts1_norm[:,0].reshape((N,1))
    A4 = (pts1_norm[:,1] * pts2_norm[:,0]).reshape((N,1))
    A5 = (pts1_norm[:,1] * pts2_norm[:,1]).reshape((N,1))
    A6 = pts1_norm[:,1].reshape((N,1))
    A7 = pts2_norm[:,0].reshape((N,1))
    A8 = pts2_norm[:,1].reshape((N,1))
    A9 = ones

    A = np.concatenate((A1, A2, A3, A4, A5, A6, A7, A8, A9), axis=1)
    U, S, V = np.linalg.svd(A)
    F = refineF(V[8,:], pts1, pts2)
    F = T.T @ F @ T

    # np.savez('q2.3_2.npz', F=F, M=M)

    return F

'''
Q2.4.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1

'''
Q2.4.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N, _ = pts1.shape
    
    Ps = np.zeros([N,4])
    for i in range(N):
        A1 = C1[0,:] - pts1[i,0]*C1[2,:] # C1_1 - x_l*C1_3
        A2 = pts1[i,1]*C1[2,:] - C1[1,:] # y_l*C1_3 - C1_2 
        A3 = C2[0,:] - pts2[i,0]*C2[2,:] # C2_1 - x_r*C2_3 
        A4 = pts2[i,1]*C2[2,:] - C2[1,:] # y_r*C2_3 - C2_2

        A = np.vstack( (A1, A2, A3, A4) )

        U, S, V = np.linalg.svd(A)
        P = V[-1, :]
        P = P/P[-1]
        Ps[i,:] = P
    
    pts1_pred = C1 @ Ps.T
    pts1_pred = (pts1_pred / pts1_pred[2,:]).transpose((1,0))
    pts2_pred = C2 @ Ps.T
    pts2_pred =(pts2_pred / pts2_pred[2,:]).transpose((1,0))
    error = np.linalg.norm(pts1_pred[:,:2] - pts1)**2 + np.linalg.norm(pts2_pred[:,:2] - pts2)**2

    return Ps, error


'''
Q2.5.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
'''
def epipolarCorrespondence(im1, im2, F, x1, y1):

    H, W, _ = im1.shape

    p_r = np.array([x1, y1, 1])
    l = F @ p_r
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s

    window_size = 5
    window_range = int(window_size/2)
    kernel = guassianFilter(window_size, 2)

    n_samples = 10
    y2s = np.arange(y1-n_samples, y1+n_samples+1)
    x2s = (-(l[1]*y2s + l[2]) / l[0]).astype(int)
    
    min_error = 1_000_000
    min_index = 0
    for i in range(n_samples):
        x2 = x2s[i]
        y2 = y2s[i]
        x1m = x1-window_range
        x1M = x1+window_range+1
        y1m = y1-window_range
        y1M = y1+window_range+1

        x2m = x2-window_range
        x2M = x2+window_range+1
        y2m = y2-window_range
        y2M = y2+window_range+1
        if x2m >= 0 and x2M <= W and y2m >= 0 and y2M <= H:
            img1_window = im1[y1m:y1M,x1m:x1M].transpose(2, 0, 1)
            img2_window = im2[y2m:y2M,x2m:x2M].transpose(2, 0, 1)
            
            error = (img1_window-img2_window).sum(axis=0)
            error = np.matmul(error, kernel)
            error = (error**2).sum()

            if error < min_error:
                min_error = error
                min_index = i

    return x2s[min_index], y2s[min_index]

def guassianFilter(size=3, sig=2):
    # print("create gaussian filter")
    idx = np.linspace(-(size - 1)/2., (size - 1)/2., size)
    x, y = np.meshgrid(idx, idx)
    kernel = np.exp(-(x**2 + y**2)) / (2*sig*sig)
    kernel /= kernel.sum()
    return kernel

'''
Q3.1: Decomposition of the essential matrix to rotation and translation.
    Input:  im1, the first image
            im2, the second image
            k1, camera intrinsic matrix of the first frame
            k1, camera intrinsic matrix of the second frame
    Output: R, rotation
            t, translation

'''
def RTRecovery(im1, im2, k1, k2):
    sift = cv.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # find best matches from stereo images
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
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
    
    E, mask = cv.findEssentialMat(src, dst, k1, cv.RANSAC, 0.99, 1.0, None)
    # R1, R2, t1 = cv.decomposeEssentialMat(E)
    # F = eightpoint(src, dst, max(im1.shape))
    # E = essentialMatrix(F, k1, k2)

    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]])
    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    # Construct 4 possible extrinsic matrix
    R1 = U @ W @ V
    R2 = U @ W.T @ V
    t1 = U[:,-1].reshape([-1, 1])/U[2,-1]
    t2 = -t1

    Rts = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    Rt1 = np.hstack((R1, t1)) 
    Rt2 = np.hstack((R1, t2)) 
    Rt3 = np.hstack((R2, t1)) 
    Rt4 = np.hstack((R2, t2))
    
    m1 = np.zeros((3, 4))
    m1[0,0] = m1[1,1] = m1[2,2] = 1

    C1 = k1 @ m1
    C2s = [k2 @ Rt1, k2 @ Rt2, k2 @ Rt3, k2 @ Rt4]

    num_pos_z = np.zeros((4,))
    for i in range(4):
        P, _ = triangulate(C1, src, C2s[i], dst)
        pos_z = (P[:,2]>0).sum()
        num_pos_z[i] = pos_z
    max_idx = num_pos_z.argmax()
    sort_num = num_pos_z.argsort()
    # print(num_pos_z)
    # print(sort_num)

    r1 = Rts[sort_num[-1]][0]
    r2 = Rts[sort_num[-2]][0]
    if (np.diag(r1) > 0).all():
        R = r1
    elif (np.diag(r2) > 0).all():
        R = r2

    _, t = Rts[sort_num[-2]]

    return R, t

'''
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        
'''

def visualOdometry(datafolder, GT_Pose, plot=True):
    xs = np.zeros((1591,1))
    ys = np.zeros((1591,1))
    zs = np.zeros((1591,1))
    for i in range(len(GT_Pose)):
        t = GT_Pose[i]
        xs[i] = t[0][3]
        ys[i] = t[1][3]
        zs[i] = t[2][3]
    path_gt = np.hstack((xs, ys, zs))

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

    k1 = np.array([K['K1']]).reshape((3,3))
    k2 = np.array([K['K2']]).reshape((3,3))

    imgs = []
    # Load Monocular images
    for file in tqdm(sorted(os.listdir(datafolder))):
        img_path = os.path.join(datafolder, file)
        # print(img_path)
        imgs.append(cv.imread(img_path))

    R_init = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1]])
    t_init = np.array([[0], [0], [0]])

    Rts = []
    Rts.append((R_init, t_init))

    # Get estimated trajectory 
    R = R_init
    t = np.array([ [GT_Pose[0][0][3]], [GT_Pose[0][1][3]], [GT_Pose[0][2][3]] ])
    cam_pos = []
    cam_pos.append(t)
    for i in tqdm(range(1, len(imgs)-1), unit='pose'): #len(imgs)
        R_cur, t_cur = RTRecovery(imgs[i-1], imgs[i], k1, k2)
        Rts.append((R_cur, t_cur))
        s = getAbsoluteScale(path_gt[i-1], path_gt[i])
        pos_cur = cam_pos[i-1] + s*(R @ t_cur)
        cam_pos.append(pos_cur)
        R = R_cur.dot(R)

    if plot:
        path_pred = np.squeeze(cam_pos)
        print(path_pred)
        np.savez('q3_2.npz', traj=path_pred)

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.plot(path_gt[:,0], path_gt[:,2], color='blue', label='gt')
        ax.plot(path_pred[:,0], -path_pred[:,2], color='red', label='pred')

        ax.legend(loc='upper left')
        plt.show()

    return cam_pos