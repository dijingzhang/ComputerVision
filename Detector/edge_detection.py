import numpy as np


class EdgeDectection():
    """
        Edge detection to grayscale images
    """

    def __init__(self, img):
        self.img = img

    def robert(self):
        img_robert = np.zeros_like(self.img)
        r, c = self.img.shape
        operator = np.array([[-1, -1], [1, 1]])
        for i in range(r):
            for j in range(c):
                if (j + 2 <= c) and (i + 2 <= r):
                    imgChild = self.img[i: i + 2, j: j + 2]
                    list_robert = operator * imgChild
                    img_robert[i, j] = abs(list_robert.sum())
        return img_robert

    def sobel(self):
        img_sobel = np.zeros_like(self.img)
        x_axis = np.zeros_like(self.img)
        y_axis = np.zeros_like(self.img)
        r, c = self.img.shape
        operator_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        operator_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        for i in range(r - 2):
            for j in range(c - 2):
                x_axis[i + 1, j + 1] = abs(np.sum(self.img[i:i + 3, j: j + 3] * operator_x))
                y_axis[i + 1, j + 1] = abs(np.sum(self.img[i:i + 3, j: j + 3] * operator_y))

                img_sobel[i + 1, j + 1] = (x_axis[i + 1, j + 1] ** 2 + y_axis[i + 1, j + 1] ** 2) ** 0.5
                # To improve the efficiency, we can replace square root with abs sum
                # img_sobel[i + 1, j + 1] = x_axis[i + 1, j + 1] + y_axis[i + 1, j + 1]

        return np.uint8(img_sobel)

    def laplace(self):
        img_laplace = np.zeros_like(self.img)
        r, c = self.img.shape
        operator = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        # operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        for i in range(r - 2):
            for j in range(c - 2):
                img_laplace[i + 1, j + 1] = abs(np.sum(self.img[i:i+3, j:j+3] * operator))
        return np.uint8(img_laplace)