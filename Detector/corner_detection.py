import numpy as np
from numpy.linalg import det, lstsq, norm
import cv2
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key


class CornerDetection():
    """
        Corner detection to grayscale images
    """
    def __init__(self, img):
        self.img = img

    def harris(self, k=0.04, th=0.1):
        Ixx, Iyy, Ixy = self._sobel_filtering(self.img)
        # H = [[Ixx, Ixy],[Ixy, Iyy]]
        # R = det(H) - k(trace(H)) ^ 2
        R = (Ixx * Iyy - Ixy ** 2) - k * ((Ixx + Iyy) ** 2)

        # detect corner
        img_harris = np.where(R >= np.max(R) * th, 255, 0)

        return np.uint8(img_harris)

    def _sobel_filtering(self, img):
        r, c = img.shape
        # sobel kernel
        sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        sobel_y = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

        # padding
        tmp = np.pad(img, (1, 1), 'edge')

        Ix, Iy = np.zeros_like(img, dtype=np.float), np.zeros_like(img, dtype=np.float32)

        # get differential
        for i in range(r):
            for j in range(c):
                Ix[i, j] = np.mean(tmp[i:i + 3, j:j + 3] * sobel_x)
                Iy[i, j] = np.mean(tmp[i:i + 3, j:j + 3] * sobel_y)
        Ix_square = Ix ** 2
        Iy_square = Iy ** 2
        Ixy = Ix * Iy

        return Ix_square, Iy_square, Ixy

    def _gaussian_filtering(self, I, K_size=3, sigma=3):
        r, c = I.shape

        # gaussian
        I_t = np.pad(I, (K_size // 2, K_size // 2), 'edge')

        # gaussian kernel
        K = np.zeros((K_size, K_size), dtype=np.float32)
        for i in range(K_size):
            for j in range(K_size):
                _x = i - K_size // 2
                _y = j - K_size // 2
                K[i, j] = np.exp(- (_x ** 2 + _y ** 2) / (2 * (sigma ** 2)))
        K /= (sigma * np.sqrt(2 * np.pi))
        K /= K.sum()

        # filtering
        for i in range(r):
            for j in range(c):
                I[i, j] = np.sum(I_t[i: i + K_size, j : j + K_size] * K)
        return I

    def sift(self, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
        """
        Refer to github: https://github.com/rmislam/PythonSIFT/blob/master/pysift.py
        """
        image = self.img.astype('float32')
        base_image = self._generateBaseImage(image, sigma, assumed_blur)
        num_octaves = self._computeNumberOfOctaves(base_image.shape)
        gaussian_kernels = self._generateGaussianKernels(sigma, num_intervals)
        gaussian_images = self._generateGaussianImages(base_image, num_octaves, gaussian_kernels)
        dog_images = self._generateDoGImages(gaussian_images)
        keypoints = self._findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
        keypoints = self._removeDuplicateKeypoints(keypoints)
        keypoints = self._convertKeypointsToInputImageSize(keypoints)

    #########################
    # Image pyramid related #
    #########################
    def _generateBaseImage(self, image, sigma, assumed_blur):
        """
        Generate base image from input image by upsampling by 2 in both direction and blurring
        """
        image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sigma_diff = np.sqrt(np.max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
        return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur

    def _computeNumberOfOctaves(self, image_shape):
        """
        Compute the number of octaves in image pyramid as function of base image shape
        """
        return np.int(np.round(np.log(min(image_shape)) / np.log(2) - 1))

    def _generateGaussianKernels(self, sigma, num_intervals):
        """
        Generate list of gaussian kernels at which to blur the input image.
        """
        num_images_per_octave = num_intervals + 3
        k = 2 ** (1. / num_intervals)
        gaussian_kernels = np.zeros(num_images_per_octave)
        gaussian_kernels[0] = sigma

        for image_index in range(1, num_images_per_octave):
            sigma_previous = (k ** (image_index - 1)) * sigma
            sigma_total = k * sigma_previous
            gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        return gaussian_kernels

    def _generateGaussianImages(self, image, num_octaves, gaussian_kernels):
        """
        Generate scale-space pyramid of Gaussian Images
        """
        gaussian_images = []

        for octave_index in range(num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
            for gaussian_kernel in gaussian_kernels[1:]:
                image = GaussianBlur(image, (0, 0), sigmaX = gaussian_kernel, sigmaY=gaussian_kernel)
                gaussian_images_in_octave.append(image)
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-3]
            image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
        return np.array(gaussian_images)

    def _generateDoGImages(self, gaussian_images):
        """
        Generate Difference-of-Gaussian image pyramid
        """
        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(subtract(second_image, first_image))
            dog_images.append(dog_images_in_octave)
        return np.array(dog_images)

    ###############################
    # Scale-space extrema related #
    ###############################
    def _findScaleSpaceExtrema(self,gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
        """
        Find pixel positions of all scale-space extrema in the image pyramid
        """
        threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)
        keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for image_index, (first_image, second_image, third_image) in enumerate(
                    zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
                # (i, j) is the center of the 3x3 array
                for i in range(image_border_width, first_image.shape[0] - image_border_width):
                    for j in range(image_border_width, first_image.shape[1] - image_border_width):
                        if self._isPixelAnExtremum(first_image[i - 1:i + 2, j - 1:j + 2],
                                             second_image[i - 1:i + 2, j - 1:j + 2],
                                             third_image[i - 1:i + 2, j - 1:j + 2], threshold):
                            localization_result = self._localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index,
                                                                                  num_intervals, dog_images_in_octave,
                                                                                  sigma, contrast_threshold,
                                                                                  image_border_width)
                            if localization_result is not None:
                                keypoint, localized_image_index = localization_result
                                keypoints_with_orientations = self._computeKeypointsWithOrientations(keypoint, octave_index,
                                                                                               gaussian_images[octave_index]
                                                                                               [localized_image_index])
                                for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
        return keypoints

    def _isPixelAnExtremum(self, first_subimage, second_subimage, third_subimage, threshold):
        """
        Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors
        False otherwise
        """
        center_pixel_value = second_subimage[1, 1]
        if np.abs(center_pixel_value) > threshold:
            if center_pixel_value > 0:
                return np.all(center_pixel_value >= first_subimage) and \
                       np.all(center_pixel_value >= third_subimage) and \
                       np.all(center_pixel_value >= second_subimage[0, :]) and \
                       np.all(center_pixel_value >= second_subimage[2, :]) and \
                       center_pixel_value >= second_subimage[1, 0] and \
                       center_pixel_value >= second_subimage[1, 2]
            elif center_pixel_value < 0:
                return np.all(center_pixel_value <= first_subimage) and \
                       np.all(center_pixel_value <= third_subimage) and \
                       np.all(center_pixel_value <= second_subimage[0, :]) and \
                       np.all(center_pixel_value <= second_subimage[2, :]) and \
                       center_pixel_value <= second_subimage[1, 0] and \
                       center_pixel_value <= second_subimage[1, 2]
        return False

    def _localizeExtremumViaQuadraticFit(self, i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma,
                                        contrast_threshold, image_border_width, eigenvalue_ratio=10,
                                        num_attempts_until_convergence=5):
        """
        Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
        """
        extremum_is_outside_image = False
        image_shape = dog_images_in_octave[0].shape
        for attempt_index in range(num_attempts_until_convergence):
            # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
            first_image, second_image, third_image = dog_images_in_octave[image_index - 1:image_index + 2]
            pixel_cube = np.stack([first_image[i - 1:i + 2, j - 1:j + 2],
                                second_image[i - 1:i + 2, j - 1:j + 2],
                                third_image[i - 1:i + 2, j - 1:j + 2]]).astype('float32') / 255.
            gradient = self._computeGradientAtCenterPixel(pixel_cube)
            hessian = self._computeHessianAtCenterPixel(pixel_cube)
            extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
            if np.abs(extremum_update[0]) < 0.5 and np.abs(extremum_update[1]) < 0.5 and np.abs(extremum_update[2]) < 0.5:
                break
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the image
            if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= \
                    image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
                extremum_is_outside_image = True
                break
            if extremum_is_outside_image:
                return None
            if attempt_index >= num_attempts_until_convergence - 1:
                return None

        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if np.abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
            xy_hessian = np.hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < (
                    (eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                # Contrast check passed -- construct and return OpenCV KeyPoint object
                keypoint = KeyPoint()
                keypoint.pt = (
                (j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + image_index * (2 ** 8) + int(
                    np.round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(num_intervals))) * (
                            2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
                keypoint.response = np.abs(functionValueAtUpdatedExtremum)
                return keypoint, image_index
        return None

    def _computeGradientAtCenterPixel(self, pixel_array):
        """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
        # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
        dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
        ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
        return np.array([dx, dy, ds])

    def _computeHessianAtCenterPixel(self, pixel_array):
        """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
        """
        # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
        # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
        # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
        # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
        # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
        center_pixel_value = pixel_array[1, 1, 1]
        dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
        dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
        dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
        dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
        dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
        dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
        return np.array([[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]])

    #########################
    # Keypoint orientations #
    #########################

    def _computeKeypointsWithOrientations(self, keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36,
                                         peak_ratio=0.8, scale_factor=1.5):
        """Compute orientations for each keypoint
        """
        keypoints_with_orientations = []
        image_shape = gaussian_image.shape

        scale = scale_factor * keypoint.size / np.float32(
            2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)
        raw_histogram = np.zeros(num_bins)
        smooth_histogram = np.zeros(num_bins)

        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_shape[0] - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_shape[1] - 1:
                        dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                        dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (
                                    i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (
                        raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] +
                                   raw_histogram[(n + 2) % num_bins]) / 16.
        orientation_max = max(smooth_histogram)
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > \
                                                    np.roll(smooth_histogram, -1)))[0]
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            if peak_value >= peak_ratio * orientation_max:
                # Quadratic peak interpolation
                # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
                left_value = smooth_histogram[(peak_index - 1) % num_bins]
                right_value = smooth_histogram[(peak_index + 1) % num_bins]
                interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                            left_value - 2 * peak_value + right_value)) % num_bins
                orientation = 360. - interpolated_peak_index * 360. / num_bins
                if abs(orientation - 360.) < np.float_tolerance:
                    orientation = 0
                new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
                keypoints_with_orientations.append(new_keypoint)
        return keypoints_with_orientations

    ##############################
    # Duplicate keypoint removal #
    ##############################

    def _compareKeypoints(self, keypoint1, keypoint2):
        """Return True if keypoint1 is less than keypoint2
        """
        if keypoint1.pt[0] != keypoint2.pt[0]:
            return keypoint1.pt[0] - keypoint2.pt[0]
        if keypoint1.pt[1] != keypoint2.pt[1]:
            return keypoint1.pt[1] - keypoint2.pt[1]
        if keypoint1.size != keypoint2.size:
            return keypoint2.size - keypoint1.size
        if keypoint1.angle != keypoint2.angle:
            return keypoint1.angle - keypoint2.angle
        if keypoint1.response != keypoint2.response:
            return keypoint2.response - keypoint1.response
        if keypoint1.octave != keypoint2.octave:
            return keypoint2.octave - keypoint1.octave
        return keypoint2.class_id - keypoint1.class_id

    def _removeDuplicateKeypoints(self, keypoints):
        """Sort keypoints and remove duplicate keypoints
        """
        if len(keypoints) < 2:
            return keypoints

        keypoints.sort(key=cmp_to_key(self._compareKeypoints))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
                    last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
                    last_unique_keypoint.size != next_keypoint.size or \
                    last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    #############################
    # Keypoint scale conversion #
    #############################

    def _convertKeypointsToInputImageSize(self, keypoints):
        """Convert keypoint point, size, and octave to input image size
        """
        converted_keypoints = []
        for keypoint in keypoints:
            keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
            keypoint.size *= 0.5
            keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
            converted_keypoints.append(keypoint)
        return converted_keypoints

    def fast(self, NonmaxSuppression=True):
        """
        Use built-in opencv function
        """
        fast = cv2.FastFeatureDetector_create()
        key_points = fast.detect(self.img, None)
        img_fast = cv2.drawKeypoints(self.img, key_points, None, color=(255, 0))

        if not NonmaxSuppression:
            fast.setNonmaxSuppression(False)
            key_points = fast.dect(self.img, None)
            img_fast = cv2.drawKeypoints(self.img, key_points, None, color=(255, 0))

        return img_fast
