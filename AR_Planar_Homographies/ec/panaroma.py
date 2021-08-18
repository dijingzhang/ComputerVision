import numpy as np
import cv2
#Import necessary functions


def matchPics(I1, I2):
    # I1, I2 : Images to match
    # use ORB to detect and describe the two images
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(I1, None)
    kp2, des2 = orb.detectAndCompute(I2, None)

    # use bf match to match keypoints
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches, kp1, kp2

def sortKP(I1, I2):
    matches, kp1, kp2 = matchPics(I1, I2)
    kp1_sorted = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    kp2_sorted = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return kp1_sorted, kp2_sorted

def main():
    left_img = cv2.imread('my_left.jpg')
    right_img = cv2.imread('my_right.jpg')
    kp1, kp2 = sortKP(left_img, right_img)
    H, _ = cv2.findHomography(kp1, kp2, cv2.RANSAC, 5.0)

    m1, n1 = left_img.shape[0], left_img.shape[1]
    m2, n2 = right_img.shape[0], right_img.shape[1]
    r_warped = cv2.warpPerspective(right_img, np.linalg.inv(H), dsize=(n1+n2, m2))
    # directly add the left_img to r_warped
    # direct = r_warped.copy()
    # direct[0: m1, 0: n1] = left_img

    # detect the most left column of left image, where two images warped
    for col in range(0, n1):
        if left_img[:, col].any() and r_warped[:, col].any():
            left = col
            break

    # detect the most right column of right image, where two images warped
    for col in range(n1-1, 0, -1):
        if left_img[:, col].any() and r_warped[:, col].any():
            right = col
            break

    res = np.zeros([m1, n1, 3], dtype=np.uint8)
    for row in range(0, m1):
        for col in range(0, n1):
            srcImgLen = float(abs(col - left))
            testImgLen = float(abs(col - right))
            alpha = srcImgLen / (srcImgLen + testImgLen)
            res[row, col] = np.clip(left_img[row, col] * (1 - alpha) + r_warped[row, col] * alpha, 0, 255)

    r_warped[0: m1, 0: n1] = res
    cv2.imshow('panorma', r_warped)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()


#Write script for Q4.2x
