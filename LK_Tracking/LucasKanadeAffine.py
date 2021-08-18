import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3)
    img_H, img_W = It.shape[0], It.shape[1]
    N = img_H * img_W

    # create a matrix with image index in homogeneous coordinates
    P_homo = np.ones((3, N))
    for i in range(img_H):
        for j in range(img_W):
            P_homo[0:2, i * img_W + j] = [i, j]

    It_Spline = RectBivariateSpline(x=np.arange(It.shape[0]), y=np.arange(It.shape[1]), z=It)
    It1_Spline = RectBivariateSpline(x=np.arange(It1.shape[0]), y=np.arange(It1.shape[1]), z=It1)

    for _ in range(int(num_iters)):
        P_index = (np.delete(arr=P_homo, obj=2, axis=0)).T
        P_index1 = (np.delete(arr=M @ P_homo, obj=2, axis=0)).T

        # find the index which exceed the region of image
        I_x = np.union1d(np.where(P_index1[:, 0] < 0), np.where(P_index1[:, 0] > img_H - 1))
        I_y = np.union1d(np.where(P_index1[:, 1] < 0), np.where(P_index1[:, 1] > img_W - 1))
        Index_delete = np.union1d(I_x, I_y)

        # remove the outside index in It and It1
        Common_part = np.delete(arr=P_index, obj=Index_delete, axis=0)
        Common_part1 = np.delete(arr=P_index1, obj=Index_delete, axis=0)

        Total = Common_part.shape[0]

        I_delta = np.zeros((Total, 2))
        A = np.zeros((Total, 6))

        Template = RectBivariateSpline.ev(self=It_Spline, xi=Common_part.T[0], yi=Common_part.T[1])
        It1_warped = RectBivariateSpline.ev(self=It1_Spline, xi=Common_part1.T[0], yi=Common_part1.T[1])
        b = Template - It1_warped

        I_delta[:, 0] = RectBivariateSpline.ev(self=It1_Spline, xi=Common_part.T[0], yi=Common_part.T[1], dx=1)
        I_delta[:, 1] = RectBivariateSpline.ev(self=It1_Spline, xi=Common_part.T[0], yi=Common_part.T[1], dy=1)

        for i in range(Total):
            w_der = np.array([[Common_part[i, 0], Common_part[i, 1], 1, 0, 0, 0],
                              [0, 0, 0, Common_part[i, 0], Common_part[i, 1], 1]])
            A[i, :] = I_delta[i, :] @ w_der

        H = A.T @ A
        p_delta = np.matmul(np.linalg.inv(H), A.T @ b)

        M[0][0] += p_delta[0]
        M[0][1] += p_delta[1]
        M[0][2] += p_delta[2]
        M[1][0] += p_delta[3]
        M[1][1] += p_delta[4]
        M[1][2] += p_delta[5]

        if np.linalg.norm(p_delta) <= threshold:
            break
    return M

