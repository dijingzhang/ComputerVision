import numpy as np
import cv2
#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from opts import get_opts
from planarH import computeH_ransac, compositeH

opts = get_opts()

book_path = '../data/book.mov'
ar_path = '../data/ar_source.mov'

frames_book = loadVid(book_path)
frames_ar = loadVid(ar_path)
cv_cover = cv2.imread('../data/cv_cover.jpg')

# remove the black region
frames_ar0 = frames_ar[0, :, :, :]
ht = frames_ar0.shape[0]
index1 = 0
for i in range(ht):
    if frames_ar0[i, 0, 0] <= 4:
        index1 += 1
    else:
        break

num_frames_book = frames_book.shape[0]
num_frames_ar = frames_ar.shape[0]
frames_book = frames_book[0: num_frames_ar, :, :, :]

m, n = cv_cover.shape[0], cv_cover.shape[1]
ratio = n / m
ar_ht = frames_ar0.shape[0] - 2 * index1
ar_wd = int(ratio * ar_ht)
begin = frames_ar0.shape[1]//2 - ar_wd // 2


fourcc = cv2.VideoWriter_fourcc(*'XVID')
w, h = frames_book.shape[2], frames_book.shape[1]
video_writer = cv2.VideoWriter('../result/ar_source.avi', fourcc, 25.0, (w, h))

for i in range(num_frames_ar):
    ar_cropped = frames_ar[i][index1: ht-index1, begin:begin+ar_wd, :]
    ar_resized = cv2.resize(ar_cropped, dsize=(n, m))
    book = frames_book[i]
    matches, locs1, locs2 = matchPics(cv_cover, book, opts)
    locs1_sorted, locs2_sorted = locs1[matches[:, 0], :], locs2[matches[:, 1], :]
    H, inliers = computeH_ransac(locs1_sorted[:, [1, 0]], locs2_sorted[:, [1, 0]], opts)
    composite_img = compositeH(H, ar_resized, book)
    video_writer.write(composite_img)
video_writer.release()


