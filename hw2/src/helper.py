"""
Homework2.
Helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import submission as sub

def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]
    return e1, e2

def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)
    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = x
        yc = y
        v = np.array([xc, yc, 1])
        l = F.dot(v) # epipolar line
        s = np.sqrt(l[0]**2+l[1]**2) # the length of epipolar line

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l/s # normalized the line
        print(f"epipolar line : {l}")

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2)
        print(f"ys, xs : {[ys, xs]} and ye, xe : {[ye, xe]}")
        
        # plot only inlier line
        xs = max(0, min(xs, sx-1))
        xe = max(0, min(xe, sx-1))
        
        ax1.plot(x, y, '*', markersize=3, linewidth=1)
        ax2.plot([xs, xe], [ys, ye], linewidth=1)
        plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F

def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000, disp=False
    )
    return _singularize(f.reshape([3, 3]))

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    # print(f"U: {U.shape} and S: {S.shape} and V: {V.shape}")
    uwv = U.dot(W).dot(V)
    # print(f"uwv : {uwv}")
    u = U[:,2].reshape([-1, 1])
    # print(f"u after reshape: {u}")

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s

def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    n_points = 0
    save_pts1 = np.empty([n_points, 2])
    save_pts2 = np.empty([n_points, 2])
    while n_points < 20:
        n_points += 1
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2)
        ax1.plot(x, y, '*', markersize=4, linewidth=2)
        # ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = sub.epipolarCorrespondence(I1, I2, F, xc, yc)

        if x2>0 and x2<I1.shape[1] and y2>0 and y2<I1.shape[0]:
            ax2.plot(x2, y2, 'ro', markersize=4, linewidth=2)
        
        save_pts1 = np.append(save_pts1, [[x, y]], axis=0)
        save_pts2 = np.append(save_pts2, [[x2, y2]], axis=0)

        plt.draw()
        
    plt.savefig('results_q2.5_1.png')
    return save_pts1, save_pts2

def getAbsoluteScale(pos_frame_prev, pose_frame_curr):
    
    """
    Estimation of scale for multiplying translation vectors
    :return: Scalar multiplier
    pos_frame_prev: the position (i.e., absolute translation) of the previous frame of GT
    pos_frame_curr: the position (i.e., absolute translation) of the current frame of GT
    """
    x_prev = pos_frame_prev[0]
    y_prev = pos_frame_prev[1]
    z_prev = pos_frame_prev[2]


    x = pose_frame_curr[0]
    y = pose_frame_curr[1]
    z = pose_frame_curr[2]

    true_vect = np.array([[x], [y], [z]])
    prev_vect = np.array([[x_prev], [y_prev], [z_prev]])

    return np.linalg.norm(true_vect - prev_vect)
