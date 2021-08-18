import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.interpolate import RectBivariateSpline

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.ones(image1.shape)
    # get the affine matrix
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters) # this one is using inverse compositional method
    M = LucasKanadeAffine(image1, image2, threshold, num_iters) # this one is using LK method
    # create the index of image
    P_homo = np.ones((3, image2.shape[0]*image2.shape[1]))
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            P_homo[0:2, i * image2.shape[1] + j] = [i, j]


    It1_Spline = RectBivariateSpline(x=np.arange(image1.shape[0]), y=np.arange(image1.shape[1]), z=image1)
    It2_Spline = RectBivariateSpline(x=np.arange(image2.shape[0]), y=np.arange(image2.shape[1]), z=image2)

    P_index1 = (np.delete(arr=M @ P_homo, obj=2, axis=0)).T
    P_index2 = (np.delete(arr=P_homo, obj=2, axis=0)).T

    # find the index which exceed the region of image
    I_x = np.union1d(np.where(P_index2[:, 0] < 0), np.where(P_index2[:, 0] > image2.shape[0] - 1))
    I_y = np.union1d(np.where(P_index2[:, 1] < 0), np.where(P_index2[:, 1] > image2.shape[1] - 1))
    Index_delete = np.union1d(I_x, I_y)

    Common_part1 = np.delete(arr=P_index1, obj=Index_delete, axis=0)
    Common_part2 = np.delete(arr=P_index2, obj=Index_delete, axis=0)

    It1 = RectBivariateSpline.ev(self=It1_Spline, xi=Common_part1.T[0], yi=Common_part1.T[1])
    It2 = RectBivariateSpline.ev(self=It2_Spline, xi=Common_part2.T[0], yi=Common_part2.T[1])

    # find the moving pixel by taking the difference between It1, It2, which is the common part of image1, image2
    Diff_img = np.maximum(It2 - It1, It1 - It2) 
    index = (np.where(Diff_img > tolerance))[0]

    mask_index = (P_index2[index, :]).astype(int)

    # chance the moving pixel into zero
    mask[mask_index.T[0], mask_index.T[1]] = 0

    # make the mask erosion and dilation in order, to eliminate the noise points
    mask_erosion = binary_erosion(mask, iterations=2).astype(mask.dtype)
    mask = binary_dilation(mask_erosion, iterations=1).astype(mask_erosion.dtype)

    return mask
