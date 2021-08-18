import numpy as np
import cv2
from loadVid import loadVid
import sys
sys.path.append('../python')
from planarH import compositeH
import time


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
    # load all data
    book_path = '../data/book.mov'
    ar_path = '../data/ar_source.mov'
    frames_book = loadVid(book_path)
    frames_ar = loadVid(ar_path)
    cv_cover = cv2.imread('../data/cv_cover.jpg')

    # remove black regions
    frames_ar0 = frames_ar[0, :, :, :]
    ht = frames_ar0.shape[0]
    index1 = 0
    for i in range(ht):
        if frames_ar0[i, 0, 0] <= 4:
            index1 += 1
        else:
            break

    # crop frames into proper size
    num_frames_book = frames_book.shape[0]
    num_frames_ar = frames_ar.shape[0]
    len = min(num_frames_book, num_frames_ar)
    frames_book = frames_book[0: len, :, :, :]

    # the index about resize and crop images
    m, n = cv_cover.shape[0], cv_cover.shape[1]
    ratio = n / m
    ar_ht = frames_ar0.shape[0]
    ar_wd = int(ratio * ar_ht)
    begin = frames_ar0.shape[1] // 2 - ar_wd // 2

    start = time.time()
    pre_time = 0
    for i in range(len):
        ar_cropped = frames_ar[i][index1: ht - index1, begin:begin + ar_wd, :]
        ar_resized = cv2.resize(ar_cropped, dsize=(n, m))
        book = frames_book[i]
        kp1, kp2 = sortKP(cv_cover, book)
        H, _ = cv2.findHomography(kp2, kp1, cv2.RANSAC, 5.0)
        composite_img = compositeH(H, ar_resized, book)
        cv2.imshow('composite_img', composite_img)
        cv2.waitKey(1)
        end = time.time()
        now_time = end-start
        diff_time = now_time - pre_time
        fps = 1 / diff_time
        print('fps:',fps)
        pre_time = now_time
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


