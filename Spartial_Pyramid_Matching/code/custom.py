import numpy as np
import visual_words
import visual_recog
import os, math, multiprocessing
from PIL import Image
from os.path import join
from sklearn import cluster
from copy import copy


def random_crop(opts, img_path, size=300):
    '''
    Randomly crop the image with a (size * size) square

    [input]
    * opts        : options
    * img_path    : path of image file to read
    * size        : the side length of cropping square

    [output]
    * image       : numpy.ndarray of shape (H, W, 3F) or (H, W)
    '''
    data_dir = opts.data_dir
    img_path = join(data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    H = img.shape[0]
    W = img.shape[1]
    if H > size and W > size:
        x = np.random.randint(0, H - size)
        y = np.random.randint(0, W - size)
        image = img[x:x+size, y:y+size, :]
    else:
        x = np.random.randint(0, (H//2))
        y = np.random.randint(0, (W//2))
        image = img[x: x+(H//2), y:y+(W//2)]
    return image

def crop_and_save(opts, crop_num = 3):
    '''
    Crop the image multi-times and save them with tags

    [input]
    * opts        : options
    * crop        : the side length of cropping square

    [save]
    * img         : numpy.ndarray of shape (M * M)
    '''
    os.mkdir('../code//cropped_img')
    data_dir = opts.data_dir
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    for i in range(len(train_files)):
        for j in range(crop_num):
            img = random_crop(opts, train_files[i])
            (filepath, tempfilename) = os.path.split(train_files[i])
            # save filename
            f = open('train_files_cropped', 'a')
            f.write(filepath+'_'+tempfilename+'_'+str(j)+'\n')
            # save labels
            f = open('train_labels_cropped', 'a')
            f.write(str(train_labels[i])+'\n')
            np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(j)), img)


def flip_img(opts, img_path):
    '''
    Flip the image

    [input]
    * opts        : options
    * img_path    : path of image file to read


    [out]
    * img_arr, image_flipped   : numpy.ndarray of shape (H, W, 3F) or (H, W)
    '''
    data_dir = opts.data_dir
    img_path = join(data_dir, img_path)
    img = Image.open(img_path)
    img_arr = np.array(img).astype(np.float32) / 255
    image = img_arr.copy()
    image = img_arr.reshape(int(img_arr.size/3), 3)
    image = np.array(image[::-1])
    image_flipped = (image.reshape(img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]))[::-1]
    return img_arr, image_flipped

def rotate_img(opts, img_path):
    '''
    rotate the image 90 degrees counterclockwise

    [input]
    * opts        : options
    * img_path    : path of image file to read


    [out]
    * img_rotated   : numpy.ndarray of shape (W, H, 3F) or (W, H)
    '''
    data_dir = opts.data_dir
    img_path = join(data_dir, img_path)
    img = Image.open(img_path)
    img_arr = np.array(img).astype(np.float32) / 255
    image = img_arr.copy()
    image = img_arr.transpose(1, 0, 2)
    img_rotated = image[::-1]
    return img_rotated

def flip_and_save(opts):
    '''
    Flip the image and save them with tags

    [input]
    * opts        : options

    [save]
    * img         : numpy.ndarray of shape (H, W, 3F) or (H, W)
    '''
    os.mkdir('../code//cropped_img')
    data_dir = opts.data_dir
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    for i in range(len(train_files)):
        img, img_flipped = flip_img(opts, train_files[i])
        (filepath, tempfilename) = os.path.split(train_files[i])
        # save filename
        f = open('train_files_cropped', 'a')
        f.write(filepath+'_'+tempfilename+'_'+'0'+'\n')
        f.write(filepath+'_'+tempfilename+'_'+'1'+'\n')
        # save labels
        f = open('train_labels_cropped', 'a')
        f.write(str(train_labels[i])+'\n')
        f.write(str(train_labels[i])+'\n')
        # save the enhanced images
        np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(0)), img)
        np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(1)), img_flipped)

def flip_and_rotate_save(opts):
    '''
    Flip and rotate the image and save them with tags

    [input]
    * opts        : options

    [save]
    * img         : numpy.ndarray of shape (H, W, 3F) or (H, W)
    '''
    os.mkdir('../code//cropped_img')
    data_dir = opts.data_dir
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    for i in range(len(train_files)):
        img, img_flipped = flip_img(opts, train_files[i])
        img_rotated = rotate_img(opts, train_files[i])
        (filepath, tempfilename) = os.path.split(train_files[i])
        for j in range(3):
            # save filename
            f = open('train_files_cropped', 'a')
            f.write(filepath+'_'+tempfilename+'_'+str(j)+'\n')
            # save labels
            f = open('train_labels_cropped', 'a')
            f.write(str(train_labels[i])+'\n')
        # save the enhanced images
        np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(0)), img)
        np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(1)), img_flipped)
        np.save(join('cropped_img', filepath+'_'+tempfilename+'_'+str(2)), img_rotated)


def compute_dictionary_one_image(opts, train_file):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''
    img = np.load(join('cropped_img', train_file+'.npy'))
    filter_responses = visual_words.extract_filter_responses(opts, img)
    x_num = img.shape[0]  # the number of pixels in x-axis
    y_num = img.shape[1]  # the number of pixels in y-axis
    filter_scales = opts.filter_scales
    feat_dir = opts.feat_dir
    alpha = opts.alpha
    f = 4 * len(filter_scales)

    #  randomly choose the index of pixels
    index_x = np.random.randint(0, x_num, size=alpha)
    index_y = np.random.randint(0, y_num, size=alpha)
    filter_responses_subset = np.zeros((alpha, 3 * f))

    #  choose alpha pixels and construct array with size [alpha, 3f]
    for i in range(alpha):
        filter_responses_subset[i, :] = filter_responses[index_x[i], index_y[i], :]

    np.save(join(feat_dir, train_file), filter_responses_subset)


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    filter_scales = opts.filter_scales

    #  build feature folder and output folder
    #  ps: if these folders have existed already, deactivate the two lines code
    # os.mkdir(feat_dir)
    # os.mkdir(out_dir)

    f = 4 * len(filter_scales)

    # load image files
    train_files = open('train_files_cropped').read().splitlines()

    # multiprocessing by calling compute_dictionary_one_image
    pool = multiprocessing.Pool(processes=n_worker)
    for train_file in train_files:
        pool.apply_async(compute_dictionary_one_image, args=(opts, train_file,))
    pool.close()
    pool.join()

    # load the temporary files back and collect the filter responses
    filter_responses = np.empty(shape=[0, 3 * f])
    for train_file in train_files:
        chosen_pixels = np.load(join(feat_dir, train_file+'.npy'))
        filter_responses = np.append(filter_responses, chosen_pixels, axis=0)

    # k-means cluster
    kmeans = cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

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


    # os.mkdir('../code//get_image_feature')
    img = np.load(join('cropped_img', img_path+'.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = visual_recog.get_feature_from_wordmap_SPM(opts, wordmap)

    # save features for good order
    np.save(join('get_image_feature', img_path), feature)


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

    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K
    M = K * (pow(4, SPM_layer_num) - 1) // 3

    features = np.empty((0, M))
    train_files = open('train_files_cropped').read().splitlines()
    train_labels = np.loadtxt('train_labels_cropped', np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    #  using multiprocessing to compute the features of all images
    pool = multiprocessing.Pool(processes=n_worker)
    for train_file in train_files:
        pool.apply_async(get_image_feature, args=(opts, train_file, dictionary,))
    pool.close()
    pool.join()

    # load these computed features back and stack them into features(N,M)
    for i in range(len(train_files)):
        chosen_feat = (np.load(join('get_image_feature', train_files[i]+'.npy'))).reshape((1, M))
        features = np.append(features, chosen_feat, axis=0)

    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
                        features=features,
                        labels=train_labels,
                        dictionary=dictionary,
                        SPM_layer_num=SPM_layer_num,
    )


# def distance_to_set(test_file, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms
    using cosine similarity

    [input]
    * test_file: the image's path for test
    * histograms: numpy.ndarray of shape (N,K)

    [save]
    * sim: numpy.ndarray of shape (N)
    '''

    # (filepath, tempfilename) = os.path.split(test_file)
    # word_hist = np.load(join('get_image_feature', filepath + '_' + tempfilename + '.npy'))
    # N = histograms.shape[0]

    # sim = np.zeros((1, N))
    # for i in range(N):
        # num = float(word_hist.dot(histograms[i].T))
        # denom = np.linalg.norm(word_hist) * np.linalg.norm(histograms[i])
        # cos = num / denom
        # sim[0, i] = 1 - (0.5 + 0.5 * cos)

    # save sim for good order to compute
    # os.mkdir(../code//sim)
    # np.save(join('sim', filepath + '_' + tempfilename), sim)









