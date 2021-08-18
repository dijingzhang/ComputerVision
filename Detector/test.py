import cv2
from edge_detection import EdgeDectection
from corner_detection import CornerDetection

# TODO: Test Edge Detection
# img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image', img)
# Operator = EdgeDectection(img)
#
# # Robert
# img_robert = Operator.robert()
# cv2.imshow('image_robert', img_robert)
#
# # Sobel
# img_sobel = Operator.sobel()
# cv2.imshow('image_sobel', img_sobel)
#
# # Laplace
# img_laplace = Operator.laplace()
# cv2.imshow('image_laplace', img_laplace)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# TODO: Test Corner Detection
chessboard = cv2.imread('chessboard.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', chessboard)
Detector = CornerDetection(chessboard)

# Harris
img_harris = Detector.harris()
cv2.imshow('image_harris', img_harris)

cv2.waitKey(0)
cv2.destroyAllWindows()