"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import util
import cv2
import math
from scipy.optimize import minimize
'''
Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    T = np.diag([1 / M, 1 / M, 1])  # the scale/normalized matrix
    pts1_nd = pts1 @ T[0:2, 0:2]  # normalized index matrix
    pts2_nd = pts2 @ T[0:2, 0:2]  # normalized index matrix

    A = np.ones((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        A[i, :] = (np.array([pts2_nd[i, 0], pts2_nd[i, 1], 1]).reshape((3, 1))
                   @ np.array([pts1_nd[i, 0], pts1_nd[i, 1], 1]).reshape(1, 3)).reshape(9)
    # using SVD to get the result F, the last row of VT
    U, sigma, VT = np.linalg.svd(A)
    F_initial = (VT[-1, :]).reshape(3, 3)
    F_nd = util.refineF(F_initial, pts1_nd, pts2_nd)
    F = T.T @ F_nd @ T  # unnormalized the fundamental matrix
    np.savez('../data/q2_1.npz', Fundamental_matrix=F, scale=M)
    return F

'''
Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = K2.T @ F @ K1
    return E

'''
Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    P = np.zeros((N, 3))
    err = 0
    for i in range(N):
        A = np.zeros((4, 4))
        A[0, :] = pts1[i, 0] * C1[2, :] - C1[0, :]
        A[1, :] = pts1[i, 1] * C1[2, :] - C1[1, :]
        A[2, :] = pts2[i, 0] * C2[2, :] - C2[0, :]
        A[3, :] = pts2[i, 1] * C2[2, :] - C2[1, :]
        # use SVD function to get the non-zero result of p, which is the last row of VT
        _, _, VT = np.linalg.svd(A)
        # divide by the homogenous scale, which is the last element of p
        P[i, :] = VT[-1, 0:3] / VT[-1, -1]
        x1 = C1 @ VT[-1, :]
        x1 /= x1[-1]
        x2 = C2 @ VT[-1, :]
        x2 /= x2[-1]
        err += np.linalg.norm(pts1[i, :]-x1[0:2]) ** 2 + np.linalg.norm(pts2[i, :]-x2[0:2]) ** 2

    return P, err

'''
3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    win_size = 3  # the size of the window, and it has to be odd
    kernel1d = cv2.getGaussianKernel(win_size, 0)
    kernel0 = np.repeat(kernel1d, win_size, axis=1)
    kernel2d = (1 / win_size) * (kernel0 @ kernel0.T)
    kernel = np.stack((kernel2d, kernel2d, kernel2d), axis=2)

    half_l = int((win_size - 1) / 2)

    win1 = im1[y1-half_l:y1+half_l+1, x1-half_l:x1+half_l+1, :]
    win1_weighted = win1 * kernel

    pts1 = np.array([x1, y1, 1]).reshape((3, 1))
    l = (F @ pts1) .reshape(3)

    H = im2.shape[0]
    win_dis = np.ones(H)
    threshold = 20
    for y2 in range(half_l, H-half_l-1):
        x2 = (-l[2]-l[1] * y2) // l[0]
        pts2 = np.array([x2, y2]).reshape((2, 1))
        win2 = im2[int(y2-half_l):int(y2+half_l+1), int(x2-half_l):int(x2+half_l+1), :]
        win2_weighted = win2 * kernel
        # check whether the distance between the two points is small enough, if not, ignore it.
        if np.linalg.norm(pts1[0:2]-pts2) <= threshold:
            win_dis[y2] = np.linalg.norm(win1_weighted-win2_weighted)
        else:
            continue
    y2 = np.argmin(win_dis)
    x2 = (-l[2]-l[1] * y2) // l[0]

    return x2, y2

'''
RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    N = pts1.shape[0]
    max_in = 0
    inliers = np.zeros(N, dtype='bool')
    most_in = 0.75 * N  # the number of inliers as we want
    pts1_homo = np.ones((N, 3))
    pts1_homo[:, 0:2] = pts1
    count = 0
    for i in range(nIters):
        inliers_compute = np.zeros(N, dtype='bool')
        # randomly pick 8 points from pts1 and pts2 to compute the F
        index = np.random.choice(range(N), 8, replace=False)
        p1 = pts1[index, :]
        p2 = pts2[index, :]
        # compute F using the 8 points
        T = np.diag([1 / M, 1 / M, 1])  # the scale/normalized matrix
        p1_nd = p1 @ T[0:2, 0:2]  # normalized index matrix
        p2_nd = p2 @ T[0:2, 0:2]  # normalized index matrix

        A = np.ones((p1.shape[0], 9))
        for i in range(p1.shape[0]):
            A[i, :] = (np.array([p2_nd[i, 0], p2_nd[i, 1], 1]).reshape((3, 1))
                       @ np.array([p1_nd[i, 0], p1_nd[i, 1], 1]).reshape(1, 3)).reshape(9)
        # using SVD to get the result F, the last row of VT
        U, sigma, VT = np.linalg.svd(A)
        F_initial = (VT[-1, :]).reshape(3, 3)
        # make F rank equals to two
        W, D, V = np.linalg.svd(F_initial)
        F_rank_two = W @ np.diag([D[0], D[1], 0]) @ V
        F0 = T.T @ F_rank_two @ T  # unnormalized the fundamental matrix
        # use estimated F to compute the line and then compute the distance between point 2 and the line
        line = (F0 @ pts1_homo.T).T
        divide = (line[:, 0]**2+line[:, 1]**2) ** 0.5
        dis = abs((line[:, 0] * pts2[:, 0] + line[:, 1] * pts2[:, 1] + line[:, 2]) / divide)
        index_inlier = np.where(dis <= tol)
        inliers_compute[index_inlier] = True
        if max_in < inliers_compute.sum():
            max_in = inliers_compute.sum()
            inliers = inliers_compute
        count += 1  # count the iteration times
        # check whether the number of inliers is above the required number
        if max_in >= most_in:
            break
    index = np.where(inliers==True)
    F = eightpoint(pts1[index[0], :], pts2[index[0], :], M=640)
    return F, inliers

'''
Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    if (r == 0).all() == True:
        R = np.identity(3)
    else:
        theta = np.linalg.norm(r)
        r_unit = r / theta
        r1, r2, r3 = r_unit[:, 0]
        r_unit_skew = np.array([[0, -r3, r2], [r3, 0, -r1], [-r2, r1, 0]])
        R = np.identity(3) + math.sin(theta) * r_unit_skew + (1 - math.cos(theta)) * (r_unit_skew @ r_unit_skew)
    return R

'''
Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    if (R == np.identity(3)).all():
        r = np.zeros((3, 1))
    else:
        theta = math.acos((np.trace(R)-1)/2)
        R0 = (R - R.T) / 2
        a, b, c = R0[2, 1], R0[0, 2], R0[1, 0]
        r1 = a * theta / math.sin(theta)
        r2 = b * theta / math.sin(theta)
        r3 = c * theta / math.sin(theta)
        r = np.array([r1, r2, r3]).reshape((3, 1))
    return r


'''
Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    C1 = K1 @ M1
    N = p1.shape[0]
    diff1 = np.empty((0, 2))
    diff2 = np.empty((0, 2))
    P = (x[0:3*N]).reshape((N, 3))
    for j in range(N):
        r2 = (x[N*3:N*3+3]).reshape((3, 1))
        R = rodrigues(r2)
        M2 = np.concatenate([R, (x[-3:]).reshape((3, 1))], axis=1)
        C2 = K2 @ M2
        hP = np.append(P[j, :], np.array([1]))  # 3D points in homogenous form
        # divide by the homogenous scale, which is the last element of p
        p1_hat = C1 @ hP
        p1_hat /= p1_hat[-1]
        p2_hat = C2 @ hP
        p2_hat /= p2_hat[-1]
        diff1 = np.append(diff1, (p1[j, :] - p1_hat[0:2]).reshape((1, 2)), axis=0)
        diff2 = np.append(diff2, (p2[j, :] - p2_hat[0:2]).reshape((1, 2)), axis=0)
    residuals = (np.concatenate([diff1.reshape(-1), diff2.reshape(-1)]))
    return residuals

'''
Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    N = p1.shape[0]
    R_init = M2_init[0:3, 0:3]
    r_init = invRodrigues(R_init)
    t = M2_init[:, -1]
    x_init = np.append(np.append(P_init.reshape(-1), r_init.reshape(-1)), t)
    # compute the reprojection error with initial parameters
    residuals_init = rodriguesResidual(K1, M1, p1, K2, p2, x_init)
    err_init = 0
    for i in range(2 * N):
        residuals_init = residuals_init.reshape((2 * N, 2))
        err_init += np.linalg.norm(residuals_init[i, :]) ** 2
    print('the reprojection error with initial M2 and w:', err_init)
    # use nonlinear optimization to get optimized extrinsics and 3D points
    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x) ** 2).sum()
    x_opt = minimize(func, x0=x_init, method='L-BFGS-B').x
    r = (x_opt[N * 3:N * 3 + 3]).reshape((3, 1))
    R = rodrigues(r)
    t = x_opt[-3:].reshape((3, 1))
    M2 = np.concatenate([R, t], axis=1)
    P2 = (x_opt[0:N * 3]).reshape((N, 3))
    # compute the reprojection error with initial parameters
    residuals_opt = rodriguesResidual(K1, M1, p1, K2, p2, x_opt)
    err_opt = 0
    for i in range(2 * N):
        residuals_opt = residuals_opt.reshape((2 * N, 2))
        err_opt += np.linalg.norm(residuals_opt[i, :]) ** 2
    print('the reprojection error with optimized M2 and w:', err_opt)

    return M2, P2

