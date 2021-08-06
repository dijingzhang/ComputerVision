import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.spatial
import skimage.color
from sklearn import cluster


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    # If the image is gray-scale image, duplicate them into three channels
    channels_of_img = img.ndim
    if channels_of_img == 2:
        img3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    else:
        img3d = img
    # rgb -> lab
    img3d_lab = skimage.color.rgb2lab(img3d)
    x_num = img3d_lab.shape[0]
    y_num = img3d_lab.shape[1]
    # set zero array for M x N x 3F of different filters
    len_scales = len(filter_scales)
    f = len_scales * 4
    filter_responses = np.zeros((x_num, y_num, 3 * f))
    index = 0
    for filter_scale in filter_scales:
        for i in range(3):
            filter_responses[:, :, i + index] = scipy.ndimage.gaussian_filter(img3d_lab[:, :, i], filter_scale)
            filter_responses[:, :, i + index + 3] = scipy.ndimage.gaussian_laplace(img3d_lab[:, :, i], filter_scale)
            filter_responses[:, :, i + index + 6] = scipy.ndimage.gaussian_filter(img3d_lab[:, :, i], filter_scale, (0, 1))
            filter_responses[:, :, i + index + 9] = scipy.ndimage.gaussian_filter(img3d_lab[:, :, i], filter_scale, (1, 0))
        index += 12
    return filter_responses

def compute_dictionary_one_image(opts, train_file):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    img_path = join(opts.data_dir, train_file)
    # is_file = isfile(img_path)
    # while is_file:     # want to check while the image exists ,but it takes too much time to compute
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    # extract the responses
    filter_responses = extract_filter_responses(opts, img)
    x_num = img.shape[0]  # the number of pixels in x-axis
    y_num = img.shape[1]  # the number of pixels in y-axis
    filter_scales = opts.filter_scales
    feat_dir = opts.feat_dir
    alpha = opts.alpha
    (filepath, tempfilename) = os.path.split(train_file)  # divide the filepath and the filename
    f = 4 * len(filter_scales)

    #  randomly choose the index of pixels
    index_x = np.random.randint(0, x_num, size=alpha)
    index_y = np.random.randint(0, y_num, size=alpha)
    filter_responses_subset = np.zeros((alpha, 3 * f))

    #  choose alpha pixels and construct array with size [alpha, 3f]
    for i in range(alpha):
        filter_responses_subset[i, :] = filter_responses[index_x[i], index_y[i], :]

    # save to a temporary file
    np.save(join(feat_dir, filepath + '_' + tempfilename), filter_responses_subset)


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    filter_scales = opts.filter_scales

    #  build feature folder and output folder
    #  ps: if these folders have existed already, deactivate the two lines code
    os.mkdir(feat_dir)
    os.mkdir(out_dir)

    f = 4 * len(filter_scales)

    # load image files
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()

    # multiprocessing by calling compute_dictionary_one_image
    pool = multiprocessing.Pool(processes=n_worker)
    for train_file in train_files:
        pool.apply_async(compute_dictionary_one_image, args=(opts, train_file,))
    pool.close()
    pool.join()

    # load the temporary files back and collect the filter responses
    filter_responses = np.empty(shape=[0, 3 * f])
    for train_file in train_files:
        (filepath, tempfilename) = os.path.split(train_file)
        chosen_pixels = np.load(join(feat_dir, filepath + '_' + tempfilename + '.npy'))
        filter_responses = np.append(filter_responses, chosen_pixels, axis=0)

    # k-means cluster
    kmeans = cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    filter_responses = extract_filter_responses(opts, img)
    x_num = img.shape[0]
    y_num = img.shape[1]

    wordmap = np.zeros((x_num, y_num))

    #  store pixels of filter response ( with size 1 x 3f) into an array and compute the distance with dictionary
    for i in range(y_num):
        to_compute = filter_responses[:, i, :]
        dist = scipy.spatial.distance.cdist(to_compute, dictionary, metric='euclidean')
        wordmap[:, i] = np.argmin(dist, axis=1)
    return wordmap