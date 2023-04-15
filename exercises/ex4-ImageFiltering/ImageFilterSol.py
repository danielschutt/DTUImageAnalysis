from scipy.ndimage import correlate
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.filters import median
import skimage as sk
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
# Ex 2, correlation with constant as off screen.


def show(input_img):
    io.imshow(input_img)
    io.show()


def ex2(input_img, const):
    res_img = correlate(input_img, weights, mode="constant", cval=const)
    return res_img


# Ex 3, mean filter with normal weights. Just makes stuff blurry, more or less.

def mean_filt(input_img, filterSize):
    size = filterSize
    # Two dimensional filter filled with 1
    weights = np.ones([size, size])
    # Normalize weights
    weights = weights / np.sum(weights)
    res_img = correlate(input_img, weights)
    return res_img

# ex 4, median filtering.

# ex 5, median filtering with size 5 looks about right.


def median_filt(input_img, filterSize):
    size = filterSize
    footprint = np.ones([size, size])
    med_img = median(input_img, footprint)
    return med_img

# ex 6, gaussian filtering. Very good at identifying big shapes, very bad at details. Higher sigma, higher blur.


def gaussian_filt(input_img, sigma):
    gauss_img = gaussian(input_img, sigma)
    return gauss_img

# Ex 7, playing with grayscaling and filters. Long of he short is, the median filter loses quite a bit of detail where the gaussian doesn't.


def gray(input_img):
    new_img = color.rgb2gray(input_img)
    return new_img


# Ex 8 and 9, prewitt filtering. Note that images MUST be grayscaled first.

def prewitt_filt(input_img, type="NormalPrewitt"):
    if type == "h":
        new_img = prewitt_h((input_img))
        return new_img
    elif type == "v":
        new_img = prewitt_v((input_img))
        return new_img
    else:
        print("Applying entire prewitt")
        new_img = prewitt((input_img))
        return new_img

    # Ex 10, filter an image completely to remove background and find edges. Use the isolate_elbow function on the elbow image.


def threshold_image(img_in, thres):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    im_float = img_in
    im_float[im_float > thres] = 1.0
    im_float[im_float < thres] = 0.0
    return (im_float)


def thresh(im_org, thresh):
    im_thresh = threshold_image(im_org, thresh)
    return im_thresh


def otsu(im_org):
    im_float = img_as_float(im_org)
    otsu_thresh = threshold_otsu(im_float)
    return img_as_ubyte(thresh(im_float, otsu_thresh))


def isolate_elbow(input_img):
    gauss_img = gaussian_filt(input_img, 20)
    prew = prewitt_filt(gauss_img)
    show(prew)
    ots = otsu(prew)
    show(ots)


# Homemade weights example
input_img = np.arange(25).reshape(5, 5)
# print(input_img)
weights = [[0, 1, 0],
           [1, 2, 1],
           [0, 1, 0]]
res_img = correlate(input_img, weights)

# General exerciss from here
# Directory containing data and images
in_dir = "exercises/ex4-ImageFiltering/data/"

# X-ray image
# im_name = "DTUSign1.jpg"
im_name = "ElbowCTSlice.png"
im_org = io.imread(in_dir + im_name)


show(im_org)
isolate_elbow(im_org)
