'''
Q2.4.3:
    1. Load point correspondences calculated and saved in Q2.3.1
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
from submission import triangulate

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

    # np.savez('q2.4_3.npz', M2=M2s[:,:,idx], C2=C2s[idx,:,:], P=Ps[idx])

    return M2s[:,:,max_idx], C1, C2s[max_idx,:,:], Ps

if __name__ == "__main__":
    img1 = cv.imread('data/image1.jpg')
    img2 = cv.imread('data/image2.jpg')

    print(img1.shape)

    H, W, _ = img1.shape

    sift = cv.ORB_create(3000)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
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

    np.savez('q2.3_1.npz', src=src, dst=dst)

    img_matches = cv.drawMatchesKnn(img1, kp1, img2, kp2, good,
                                    None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    # visualize(img1, img2, F, K1, K2)

    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M2, C2, P = findM2(M2s, K1, K2, src, dst)

    # np.savez('q2.4_3.npz', M2=M2, C2=C2, P=P)

