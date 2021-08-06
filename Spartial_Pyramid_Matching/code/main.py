from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts
import custom


def main():
    opts = get_opts()

    # Q1.1
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # filter_responses = visual_words.extract_filter_responses(opts, img)
    # util.display_filter_responses(opts, filter_responses)

    # Q1.2
    n_cpu = util.get_num_CPU()
    if __name__ == '__main__':
        visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    # Q1.3
    # img_path = join(opts.data_dir, 'park/labelme_aumetbzppbkuwju.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32)/255
    # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    # wordmap = visual_words.get_visual_words(opts, img, dictionary)
    # out_path = join(opts.wordmap_dir, 'park_labelme_aumetbzppbkuwju.jpg')
    # util.visualize_wordmap(wordmap, out_path=out_path)

    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    if __name__ == '__main__':
        visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    if __name__ == '__main__':
        conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
        print(conf)
        print(accuracy)
        np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
        np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

    # Q3.2 When using flip method to test, active the seond till the last code; when using crop methods,
    # deactive the second line and active the others; when using flip and rotate to create triple numbers
    # of train files, active the third line till the last

    # custom.crop_and_save(opts)  # crop the images into several small ones and use them as train files
    # custom.flip_and_save(opts) # flip the images and create a bank of train files with double numbers
    # custom.flip_and_rotate_save(opts) # flip and rotate the images and create a bank of train files with triple numbers

    # n_cpu = util.get_num_CPU()
    # if __name__ == '__main__':
        # custom.compute_dictionary(opts, n_worker=n_cpu)

    # if __name__ == '__main__':
        # custom.build_recognition_system(opts, n_worker=n_cpu)

    # if __name__ == '__main__':
        # conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
        # print(conf)
        # print(accuracy)
        # np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
        # np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')

if __name__ == '__main__':
    main()
