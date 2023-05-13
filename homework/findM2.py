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

    M1 = [ [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0] ]
    
    C1 = K1 @ M1
    C2s = np.array([K2 @ M2s[:,:,0], K2 @ M2s[:,:,1],  
                    K2 @ M2s[:,:,2], K2 @ M2s[:,:,3]])

    Ps = []
    errs = []
    n_negs = np.zeros((4,))

    for i in range(4):
        # print(f"Triangulate at {i} with {src.shape} and {dst.shape}")
        P, err = triangulate(C1, src, C2s[i,:,:], dst)
        Ps.append(P)
        errs.append(err)
        num_negs = (P<0).sum()
        n_negs[i] = num_negs
        # print(num_negs)

    min_idx = n_negs.argmin()

    # print(f"n_neg1: {n_negs[0]}, n_neg2: {n_negs[1]}, n_neg3 {n_negs[2]}, n_neg4: {n_negs[3]}")
    # print(f"smallest negative at {min_idx}")
    # np.savez('q2.4_3.npz', M2=M2s[:,:,idx], C2=C2s[idx,:,:], P=Ps[idx])

    return M2s[:,:,min_idx], C1, C2s[min_idx,:,:], Ps[min_idx], errs[min_idx]
