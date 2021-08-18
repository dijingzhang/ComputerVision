import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Put your implementation here
    x1, y1 = rect[0], rect[1]
    x2, y2 = rect[2], rect[3]
    img_h = int(y2 + 1 - y1)
    img_w = int(x2 + 1 - x1)
    N = img_h * img_w
    p = p0.copy()

    # create the template from It
    It_Spline = RectBivariateSpline(x=np.arange(It.shape[0]), y=np.arange(It.shape[1]), z=It)
    Pixel_index = np.zeros((N, 2))
    for i in range(img_h):
        for j in range(img_w):
            Pixel_index[i*img_w+j] = [y1+i, x1+j]
    Template = RectBivariateSpline.ev(self=It_Spline, xi=Pixel_index.T[0], yi=Pixel_index.T[1])

    It1_Spline = RectBivariateSpline(x=np.arange(It1.shape[0]), y=np.arange(It1.shape[1]), z=It1)

    for _ in range(int(num_iters)):
        I_delta = np.zeros((N, 2))

        I_warped = RectBivariateSpline.ev(self=It1_Spline, xi=Pixel_index.T[0]+p[1], yi=Pixel_index.T[1]+p[0])
        error = Template - I_warped
        b = error.reshape((N, 1))

        # compute the delta I, which is the derivative of warped I
        I_delta[:, 0] = RectBivariateSpline.ev(self=It1_Spline, xi=Pixel_index.T[0]+p[1],
                                               yi=Pixel_index.T[1]+p[0], dy=1)
        I_delta[:, 1] = RectBivariateSpline.ev(self=It1_Spline, xi=Pixel_index.T[0]+p[1],
                                               yi=Pixel_index.T[1]+p[0], dx=1)

        I_delta_T = np.transpose(I_delta)
        H = np.matmul(I_delta_T, I_delta)
        p_delta = np.matmul(np.linalg.inv(H), np.matmul(I_delta_T, b))
        p[0] += p_delta[0][0]
        p[1] += p_delta[1][0]
        if np.linalg.norm(p_delta) <= threshold:
            break
    return p


