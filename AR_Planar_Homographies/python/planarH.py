import numpy as np
import cv2
import math

def computeH(x1, x2):
	#Compute the homography between two sets of points
	N = x1.shape[0]
	A = np.empty((0, 1))
	for i in range(N):
		x1_x, x1_y = x1[i][0], x1[i][1]
		x2_x, x2_y = x2[i][0], x2[i][1]
		A_current = np.array([x2_x, x2_y, 1, 0, 0, 0, -x1_x*x2_x, -x1_x*x2_y, -x1_x,
				     		  0, 0, 0, x2_x, x2_y, 1, -x1_y*x2_x, -x1_y*x2_y, -x1_y])
		A = np.append(A, A_current)
	A = np.reshape(A, newshape=(2 * N, 9))
	_, _, VT = np.linalg.svd(A)
	h = VT[-1, :]
	H2to1 = h.reshape((3, 3))
	return H2to1

def computeH_norm(x1, x2):
	#Compute the centroid of the points
	N = x1.shape[0]
	M_mean = np.ones((N, N))
	M_mean *= (1/N)
	sum_x1_x = np.sum(x1[:, 0])
	sum_x1_y = np.sum(x1[:, 1])
	sum_x2_x = np.sum(x2[:, 0])
	sum_x2_y = np.sum(x2[:, 1])
	# x1_centroid = np.matmul(M_mean, x1)  # x1_centroid is a N×2 matrix
	# x2_centroid = np.matmul(M_mean, x2)  # x2_centroid is a N×2 matrix


	#Shift the origin of the points to the centroid
	I = np.identity(N)
	x1_normalized = np.matmul((I-M_mean), x1)
	x2_normalized = np.matmul((I-M_mean), x2)
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	max1, max2 = 0.0001, 0.0001
	for i in range(N):
		current1 = x1_normalized[i][0] ** 2 + x1_normalized[i][1] ** 2
		current2 = x2_normalized[i][0] ** 2 + x2_normalized[i][1] ** 2
		if current1 > max1:
			max1 = current1
		if current2 > max2:
			max2 = current2

	constant1 = 1 / math.sqrt(max1)
	constant2 = 1 / math.sqrt(max2)
	x1_normalized *= constant1
	x2_normalized *= constant2
	# Similarity transform 1
	T1 = np.array([constant1, 0, constant1 * (-sum_x1_x / N), 0, constant1, constant1 * (-sum_x1_y / N), 0, 0, 1]).reshape((3, 3))

	#Similarity transform 2
	T2 = np.array([constant2, 0, constant2 * (-sum_x2_x / N), 0, constant2, constant2 * (-sum_x2_y / N), 0, 0, 1]).reshape((3, 3))

	#Compute homography
	H = computeH(x1_normalized, x2_normalized)

	#Denormalization
	T1_inv = np.linalg.inv(T1)
	HT2 = np.matmul(H, T2)
	H2to1 = np.matmul(T1_inv, HT2)
	return H2to1

def computeH_ransac(locs1, locs2, opts):
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
	N = locs1.shape[0]
	num_chosen = 4
	inlier_max = 0
	bestH2to1 = np.zeros((3, 3))
	inliers = np.zeros((N,))
	for i in range(max_iters):
		locs1_current = np.empty((0, 1))
		locs2_current = np.empty((0, 1))
		# choose matched points randomly from locs
		index = np.random.randint(0, N, num_chosen)
		for j in range(num_chosen):
			locs1_current = np.append(locs1_current, locs1[index[j], :])
			locs2_current = np.append(locs2_current, locs2[index[j], :])
		locs1_current = np.reshape(locs1_current, newshape=(num_chosen, 2))
		locs2_current = np.reshape(locs2_current, newshape=(num_chosen, 2))
		H2to1 = computeH_norm(locs1_current, locs2_current)

		x2_homo = np.ones((3, ))
		inliers_current = np.zeros((N,))

		for k in range(N):
			x2_homo[0] = locs2[k][0]
			x2_homo[1] = locs2[k][1]
			x1_homo = np.matmul(H2to1, x2_homo)
			# scale the third element in x1 to '1'
			if x1_homo[2] != 0:
				scale = 1 / x1_homo[2]
			else:
				scale = 1
			x1_homo *= scale
			distance = (x1_homo[0] - locs1[k][0]) ** 2 + (x1_homo[1] - locs1[k][1]) ** 2
			if distance <= inlier_tol ** 2:
				inliers_current[k] = 1
		inlier_sum = np.sum(inliers_current)
		if inlier_sum > inlier_max:
			inlier_max = inlier_sum
			inliers = inliers_current
			bestH2to1 = H2to1

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.

	#Create mask of same size as template
	H = np.linalg.inv(H2to1)
	m, n = img.shape[0], img.shape[1]
	j, k = template.shape[0], template.shape[1]
	mask = 255 * np.ones([j, k], dtype=np.uint8)

	#Warp mask by appropriate homography
	mask1 = cv2.warpPerspective(mask, H, dsize=(n, m))
	mask2 = cv2.bitwise_not(mask1)

	#Warp template by appropriate homography
	template1 = cv2.warpPerspective(template, H, dsize=(n, m))
	img1 = cv2.bitwise_and(img, img, mask=mask2)
	template2 = cv2.bitwise_and(template1, template1, mask1)

	#Use mask to combine the warped template and the image
	composite_img = cv2.add(img1, template2, dtype=0)
	return composite_img

	# use template as mask, but will encounter problems when facing all '0' template
	# template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
	# _, mask1 = cv2.threshold(template_gray, 1, 255, cv2.THRESH_BINARY_INV)
	# _, mask2 = cv2.threshold(template_gray, 1, 255, cv2.THRESH_BINARY)
	# img1 = cv2.bitwise_and(img, img, mask=mask1)
	# template1 = cv2.bitwise_and(template, template, mask=mask2)
	# composite_img = cv2.add(img1, template1, dtype=0)


