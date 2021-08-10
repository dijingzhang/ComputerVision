Implement a scene classification system that uses the bag-of-words approach with its spatial pyramid extension

Pipeline:

* Extracting Filter Responses

Build up a filter bank on images (Gaussian, LoG, DoG in x and DoG in y) and will have a total of 3F filter responses per pixel if the filter bank is of size F

* Creating Visual Words

Create a dictionary of visual words from the filter responses using k-means. Instead of using all of the filter responses, we will use responses at **α random pixels**. If the size of training images is T, we will collcet a matrix **filter_responses** that is αT x 3F. Then generate a visual words dictionary with K words (K means).

* Computing Visual Words

Map each pixel in the image to its closest word in dictionary

* Extracting Features

Represent a image with a bag of words and simply look at how often each word appears (create a histogram)

* Multi-resolution: Spatial Pyramid Matching

Bag of words is simple and efficient but it discards the information about the spatial structure of the image and this information if often valuable. SPM is to divide the image into a small number of cells and concatenate the histogram of each of theses cells to the histogram of the original image with a suitable weight. Final, if there are **L+1** layers and **K** visual words, the resulting vector has dimension K(4^(L+1) - 1) / 3

* Comparing Images

Need to compare images to find the nearest instance in the training data. Use histogram intersection similarity, which is the sum of the minimum value of each corresponding bins. The largest value indicated the nearest instance. We will build histograms to all training images and it will be a matrix *T x K(4^(L+1) - 1) / 3* then we can build histogram to test image and make comparsion.
