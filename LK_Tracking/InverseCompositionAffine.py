import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    M = np.eye(3)
    N = It.shape[0] * It.shape[1]
    P_homo = np.ones((3, N))
    for i in range(It.shape[0]):
        for j in range(It.shape[1]):
            P_homo[0:2, i * It.shape[1] + j] = [i, j]
    T_index = (np.delete(arr=P_homo, obj=2, axis=0)).T

    It_Spline = RectBivariateSpline(x=np.arange(It.shape[0]), y=np.arange(It.shape[1]), z=It)
    # evalute the gradient delta T of the template T, which is It
    T_delta = np.zeros((N, 2))
    T_delta[:, 0] = RectBivariateSpline.ev(self=It_Spline, xi=T_index.T[0], yi=T_index.T[1], dx=1)
    T_delta[:, 1] = RectBivariateSpline.ev(self=It_Spline, xi=T_index.T[0], yi=T_index.T[1], dy=1)

    # evaluate the Jacobian at(x;0) and compute the steepest descent images, then we can get the Hessian matrix
    A = np.zeros((N, 6))
    for i in range(N):
        w_der = np.array([[T_index[i, 0], T_index[i, 1], 1, 0, 0, 0],
                          [0, 0, 0, T_index[i, 0], T_index[i, 1], 1]])
        A[i, :] = T_delta[i, :] @ w_der
    H = A.T @ A

    It1_Spline = RectBivariateSpline(x=np.arange(It1.shape[0]), y=np.arange(It1.shape[1]), z=It1)

    for _ in range(int(num_iters)):
        P_index1 = (np.delete(arr=M @ P_homo, obj=2, axis=0)).T

        I_x = np.union1d(np.where(P_index1[:, 0] < 0), np.where(P_index1[:, 0] > It.shape[0] - 1))
        I_y = np.union1d(np.where(P_index1[:, 1] < 0), np.where(P_index1[:, 1] > It.shape[1] - 1))
        Index_delete = np.union1d(I_x, I_y)

        # compute the warped It1 and get the error, then set the pixel value equals zero, which is not in the region
        It1_warped = RectBivariateSpline.ev(self=It1_Spline, xi=P_index1.T[0], yi=P_index1.T[1])
        b = It1_warped - It.reshape(N)
        b[Index_delete] = 0

        p_delta = np.matmul(np.linalg.inv(H), A.T @ b)
        M_delta = np.array([[1+p_delta[0], p_delta[1], p_delta[2]], [p_delta[3], 1+p_delta[4], p_delta[5]], [0, 0, 1]])
        M = M @ np.linalg.inv(M_delta)
        if np.linalg.norm(p_delta) <= threshold:
            break
    return M
