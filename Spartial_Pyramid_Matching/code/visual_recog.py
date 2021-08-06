import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words. -- bag of words
    Simple and efficient but it discards information about the spatial structure

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    hist, _ = (np.histogram(wordmap, bins=K, range=[0, K]))
    hist = hist / hist.sum()
    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.
    SPM: divide the image into a small number of cells and concatenate the histogram of each of these cells to the
         histogram of the original image, with a suitable weight

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3x)
    '''

    K = opts.K
    L = opts.L
    l_num = pow(2, L - 1)  # the number of the layers
    hist_all_2d = np.empty(shape=[0, K])

    #  divide the 2d array into l_num parts in the row direction and then in the column direction
    #  next compute the histograms of the finest parts
    wordmap_x = np.array_split(wordmap, l_num, axis=0)
    hist_finest = np.empty(shape=[0, K])
    for i in range(l_num):
        wordmap_y = np.array_split(wordmap_x[i], l_num, axis=1)
        for j in range(l_num):
            hist = get_feature_from_wordmap(opts, wordmap_y[j])
            hist_reshaped = hist.reshape((1, K))
            hist_finest = np.append(hist_finest, hist_reshaped)
    hist_finest_2d = np.reshape(hist_finest, (l_num * l_num, K))  # reshape to 2d, in oder to compute easily
    hist_all_2d = np.append(hist_all_2d, hist_finest_2d, axis=0)  # store all 2d-data to hist_all_2d

    #  aggregate from the histograms of the finer layers to get the ones of coarser layers
    #  without taking the weights into consideration
    index1 = 0
    index2 = l_num * l_num
    #  traverse and compute all the histograms from different levels
    for k in range(L - 1, 0, -1):
        l_finer_num = pow(2, k)  # the number of the finer layer
        l_coarser_num = pow(2, k - 1)  # the number of the coarser layer
        hist_finer = hist_all_2d[index1: index2]
        hist_current_2d = np.empty(shape=[0, K])

        #  traverse the finer histograms
        for m in range(l_coarser_num):
            for n in range(l_coarser_num):
                hist_sum = np.zeros((1, K))
                #  sum 2 x 2 histograms of the finer histograms
                for i in range(2):
                    for j in range(2):
                        hist_sum += hist_finer[(i + 2 * m) * l_finer_num + j + 2 * n]
                hist_sum *= 1 / 4
                hist_current_2d = np.append(hist_current_2d, hist_sum, axis=0)
        index1 = index2
        index2 += (l_coarser_num * l_coarser_num)
        hist_all_2d = np.append(hist_all_2d, hist_current_2d, axis=0)

    # multiply by the weights
    index3 = 0
    index4 = l_num * l_num
    for i in range(L, 0, -1):
        if i == 1:
            hist_all_2d[index3: index4] *= pow(2, (1 - L))
        else:
            hist_all_2d[index3: index4] *= (pow(2, (i - L - 1)) * pow(4, (1 - i)))
        index3 = index4
        index4 += pow(4, (i - 2))

    hist_all = hist_all_2d.flatten()  # flatten the 2d histograms
    return hist_all


def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [save]
    * feature: numpy.ndarray of shape (K)
    '''

    data_dir = opts.data_dir
    (filepath, tempfilename) = os.path.split(img_path)

    os.mkdir('../code//get_image_feature')
    img_path_all = join(data_dir, img_path)
    img = Image.open(img_path_all)
    img = np.array(img).astype(np.float32) / 255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    # save features for good order
    np.save(join('get_image_feature', filepath + '_' + tempfilename), feature)


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K
    M = K * (pow(4, SPM_layer_num) - 1) // 3

    features = np.empty((0, M))
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #  using multiprocessing to compute the features of all images
    pool = multiprocessing.Pool(processes=n_worker)
    for train_file in train_files:
        pool.apply_async(get_image_feature, args=(opts, train_file, dictionary,))
    pool.close()
    pool.join()

    # load these computed features back and stack them into features(N,M)
    for train_file in train_files:
        (filepath, tempfilename) = os.path.split(train_file)
        chosen_feat = (np.load(join('get_image_feature', filepath + '_' + tempfilename + '.npy'))).reshape((1, M))
        features = np.append(features, chosen_feat, axis=0)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(test_file, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.
    Histrogram intersection similarity: the sum of the minimum value of each corresponding bins

    [input]
    * test_file: the image's path for test
    * histograms: numpy.ndarray of shape (N,K)

    [save]
    * sim: numpy.ndarray of shape (N)
    '''

    (filepath, tempfilename) = os.path.split(test_file)
    word_hist = np.load(join('get_image_feature', filepath + '_' + tempfilename + '.npy'))
    compared_array = np.minimum(word_hist, histograms)
    compared_array_sum = np.sum(compared_array, axis=1)
    sim = 1 - compared_array_sum

    # save sim for good order to compute
    os.mkdir('../code//sim')
    np.save(join('sim', filepath + '_' + tempfilename), sim)

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']
    dictionary = trained_system['dictionary']

    conf = np.zeros((8, 8))

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # using multiprocessing to evaluate
    pool = multiprocessing.Pool(processes=n_worker)
    for test_file in test_files:
        pool.apply_async(get_image_feature, args=(opts, test_file, dictionary,))
    pool.close()
    pool.join()

    pool = multiprocessing.Pool(processes=n_worker)
    for test_file in test_files:
        pool.apply_async(distance_to_set, args=(test_file, trained_features,))
    pool.close()
    pool.join()

    # load similarity back to get the predicted label, and build the confusion matrix
    for i in range(len(test_files)):
        (filepath, tempfilename) = os.path.split(test_files[i])
        sim = (np.load(join('sim', filepath + '_' + tempfilename + '.npy')))
        trained_index = np.argmin(sim)
        predicted_label = trained_labels[trained_index]
        actual_label = test_labels[i]
        conf[actual_label, predicted_label] += 1

    accuracy = np.trace(conf) / np.sum(conf)
    return conf, accuracy



