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
    
    # print(f"M: {M}")
    # print(f"pts1 shape : {pts1.shape}")
    # print(f"pts2 shape : {pts2.shape}")
    
    N, _ = pts1.shape
    
    ones = np.ones((N,1))

    # m1 = pts1.mean(axis=0)
    # m2 = pts2.mean(axis=0)
    
    # m1 = np.array([ [1, 0, -m1[0]],
    #                 [0, 1, -m1[1]],
    #                 [0, 0,      0] ])

    # m2 = np.array([ [1, 0, -m2[0]],
    #                 [0, 1, -m2[1]],
    #                 [0, 0,      0] ])
    

    T = np.array([ [1.0/M,    0,   0],
                   [0,    1.0/M,   0],
                   [0,        0,   1] ])

    # T1 = M @ m1
    # T2 = M @ m2

    pts1_T = np.append(pts1, ones, axis=1).T
    pts2_T = np.append(pts2, ones, axis=1).T
    pts1_norm = (T @ pts1_T).T
    pts2_norm = (T @ pts2_T).T

    A1 = (pts1_norm[:,0] * pts2_norm[:,0]).reshape(N,1)
    A2 = (pts1_norm[:,0] * pts2_norm[:,1]).reshape(N,1)
    A3 = pts1_norm[:,0].reshape((N,1))
    A4 = (pts1_norm[:,1] * pts2_norm[:,0]).reshape(N,1)
    A5 = (pts1_norm[:,1] * pts2_norm[:,1]).reshape(N,1)
    A6 = pts1_norm[:,1].reshape(N,1)
    A7 = pts2_norm[:,0].reshape(N,1)
    A8 = pts2_norm[:,1].reshape(N,1)
    A9 = ones

    A = np.concatenate((A1, A2, A3, A4, A5, A6, A7, A8, A9), axis=1)

    # print(f"A  : {A}")

    U, S, V = np.linalg.svd(A)

    F = refineF(V[-1,:], pts1, pts2)
    
    F = T.T @ F @ T   

    # print(f"T1 : ")
    # print(T1)

    # print(f"T1.T : ")
    # print(T1.T)

    return F

'''
Q2.4.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation

    # K1_I = np.linalg.inv(K1).T
    # k2_I = np.linalg.inv(K2)

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
    # Replace pass by your implementation
    # print("At Triangulate")
    N, _ = pts1.shape
    
    # pts1 = np.append(pts1, np.ones((N,1)), axis=1)
    # pts2 = np.append(pts2, np.ones((N,1)), axis=1)
    # print(f" After append homo : {pts1.shape} and {pts2.shape}")

    Ps = np.zeros([N,4])
    for i in range(N):

        A1 = pts1[i,0]*C1[2,:] - C1[0,:] # C1_1 - x_l*C1_3
        A2 = pts1[i,1]*C1[2,:] - C1[1,:] # y_l*C1_3 - C1_2 
        A3 = pts2[i,0]*C2[2,:] - C2[0,:] # C2_1 - x_r*C2_3 
        A4 = pts2[i,1]*C2[2,:] - C2[1,:] # y_r*C2_3 - C2_2

        A = np.vstack( (A1, A2, A3, A4) )

        U, S, V = np.linalg.svd(A)
        P = V[-1, :]
        P = P/P[-1]
        # if i<10: print(P)
        Ps[i,:] = P
    
    # print(f" 3d points {Ps.shape} and {C1.shape} and {C2.shape}")

    pts1_pred = C1 @ Ps.T
    pts1_pred = (pts1_pred / pts1_pred[2,:]).transpose((1,0))
    # print(f" Ps.shape : {Ps.shape}")
    # print(f" pts1_pred : {pts1_pred.shape}")

    pts2_pred = C2 @ Ps.T
    pts2_pred =(pts2_pred / pts2_pred[2,:]).transpose((1,0))
    # print(f" pts2_pred : {pts2_pred.shape}")

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
    # Replace pass by your implementation
    # print(f"Caculate epipolar line at : {x1} and {y1}")

    H, W, _ = im1.shape

    p_r = np.array([x1, y1, 1])
    l = F @ p_r
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s

    # print(f"line : {l}")
    window_size = 5
    window_range = int(window_size/2)
    kernel = guassianFilter(window_size, 2)

    # print("kernel")
    # print(kernel)

    n_samples = 10
    y2s = np.arange(y1-n_samples, y1+n_samples+1)
    x2s = (-(l[1]*y2s + l[2]) / l[0]).astype(int)
    
    # print(f"x2s : {x2s}")
    # print(f"y2s : {y2s}")

    min_error = 1_000_000
    min_index = 0
    for i in range(n_samples):
        x2 = x2s[i]
        y2 = y2s[i]
        # print(f"At img2 (x,y): {x2}, {y2}")
        x1m = x1-window_range
        x1M = x1+window_range+1
        y1m = y1-window_range
        y1M = y1+window_range+1

        x2m = x2-window_range
        x2M = x2+window_range+1
        y2m = y2-window_range
        y2M = y2+window_range+1
        # print(f"x1m : {x1m} and x1M : {x1M}, y1m : {y1m} and y1M : {y1M}")
        # print(f"x2m : {x2m} and x2M : {x2M}, y2m : {y2m} and y2M : {y2M}")
        if x2m >= 0 and x2M <= W and y2m >= 0 and y2M <= H:
            
            img1_window = im1[y1m:y1M,x1m:x1M].transpose(2, 0, 1)
            img2_window = im2[y2m:y2M,x2m:x2M].transpose(2, 0, 1)
            
            # print(f"img1_window: ")
            # print(img1_window)


            # print(f"img2_window: ")
            # print(img2_window)

            error = (img1_window-img2_window).sum(axis=0)
            # print(f" shape of error : {error.shape}")

            error = np.matmul(error, kernel)
            error = (error**2).sum()

            if error < min_error:
                min_error = error
                min_index = i
                # print(f"Minimum error at {i} : {min_error}")
            # print(img2_window.shape)
            # print(img1_window.shape)
    
    # print(f"Min: At {min_index} :  (x2: {x2s[min_index]} and y2: {y2s[min_index]})")

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
from findM2 import *
def essentialDecomposition(im1, im2, k1, k2):
    # Replace pass by your implementation
    
    H, W, _ = im1.shape

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)

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

    # img_matches = cv.drawMatchesKnn(im1, kp1, im2, kp2, good,
    #                                 None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    F = eightpoint(src, dst, max(H, W))
    E = essentialMatrix(F, k1, k2)

    U, S, V = np.linalg.svd(E)
    W = np.array([ 
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
        ])

    # print(f"U : {U}")

    R1 = U @ W @ V.T
    R2 = U @ W.T @ V.T
    T1 = U[:,-1]
    T1 = T1[:,None]
    T2 = -T1

    # print("R1:")
    # print(R1)

    # I = np.array([ [1,0,0],
    #                [0,1,0],
    #                [0,0,1] ])    
    # t1 = np.append(I, T1, axis=1)
    # t2 = np.append(I, T2, axis=1)

    # print("t1:")
    # print(t1)
    
    # print("t2:")
    # print(t2)

    Rt1 = np.hstack((R1, T1)) #R1@t1 #
    Rt2 = np.hstack((R1, T2)) #R1@t2 #
    Rt3 = np.hstack((R2, T1)) #R2@t1 #
    Rt4 = np.hstack((R2, T2)) #R2@t2 #

    Rts = [(R1, T1), (R1, T2), (R2, T1), (R2, T2)]

    m1 = np.zeros((3, 4))
    m1[0,0] = m1[1,1] = m1[2,2] = 1
    # print(m1)

    C1 = k1 @ m1
    C2s = [k2 @ Rt1, k2 @ Rt2, k2 @ Rt3, k2 @ Rt4]

    # print("C1: ")
    # print(C1)

    # print("C2: ")
    # print(C2s)

    num_pos = np.zeros((4,))
    for i in range(4):
        P, err = triangulate(C1, src, C2s[i], dst)
        # print("3d points: ")
        # print(P.shape)
        pos_z = (P[:,2]>0).sum()
        num_pos[i] = pos_z

    # print(num_pos)
    max_idx = num_pos.argmax()
    # print(max_idx)

    R, t = Rts[max_idx]
    
    return R, t


'''
Q3.2: Implement a monocular visual odometry.
    Input:  datafolder, the folder of the provided monocular video sequence
            GT_pose, the provided ground-truth (GT) pose for each frame
            plot=True, draw the estimated and the GT camera trajectories in the same plot
    Output: trajectory, the estimated camera trajectory (with scale aligned)        

'''
def visualOdometry(datafolder, GT_Pose, plot=True):
    # Replace pass by your implementation
    
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
    
    print(len(imgs))

    Rts = []
    R_init = np.array([[1,0,0],
                       [0,1,0],
                       [0,0,1] ]
                    )
    t_init = np.array([[0], [0], [0]])

    Rts.append((R_init, t_init))

    # Accumulate estimated poses from sequence of monocular images
    for i in tqdm(range(1, len(imgs)), unit='pose'):
        im1 = imgs[i-1]
        im2 = imgs[i]
        R_curr, t_curr = essentialDecomposition(im1, im2, k1, k2)
        print("R: ")
        print(R_curr)
        print("t: ")
        print(t_curr)
        Rts.append((R_curr, t_curr))

        # print(f"s : {s}")
        # raise NotImplementedError(f"Stop at visual odometry {i}")

    # Get a trajectory 
    R_accum = R_init
    t = np.array([ [GT_Pose[0][0][3]], [GT_Pose[0][1][3]], [GT_Pose[0][2][3]] ])
    
    print("Initial pose: ")
    print(t)

    pos_pred = []
    pos_pred.append(t)

    for i in tqdm(range(1, len(Rts)), unit='trajectory'):
        R_prev, t_prev = Rts[i-1]
        R_curr, t_curr = Rts[i]
        print(R_accum.shape)
        R_accum = R_accum @ R_curr 
        s = getAbsoluteScale(t_curr, t_prev)
        
        t_pred = pos_pred[i-1] + s*(R_accum @ t_curr)
        pos_pred.append(t_pred)

    return pos_pred



