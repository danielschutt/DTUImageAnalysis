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
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk


def show(input_img):
    io.imshow(input_img)
    io.show()

# convenience function for showing plots side by side


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()


def gray(input_img):
    new_img = color.rgb2gray(input_img)
    return new_img


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

# Ex 1, otsu thresholding comparison

# Ex 2, erosion. Good for making elements smaller and not touch each other. Removes holes.


def ero(im_org):
    footprint = disk(1)
# Check the size and shape of the structuring element
    print(footprint)
    bin_img = otsu(gray(im_org))
    eroded = erosion(bin_img, footprint)
    plot_comparison(bin_img, eroded, 'erosion')

# Ex 3, dilation. Good for making elements bigger, and enlargening holes.


def dil(im_org):
    footprint = disk(1)
# Check the size and shape of the structuring element
    print(footprint)
    bin_img = otsu(gray(im_org))
    eroded = dilation(bin_img, footprint)
    plot_comparison(bin_img, eroded, 'erosion')

    # Ex 4, opening and closing. Opening removes small objects but keeps original size. Closing fills holes but keeps original size.


def open_img(im_org):
    footprint = disk(1)
# Check the size and shape of the structuring element
    print(footprint)
    bin_img = otsu(gray(im_org))

    opened = opening(bin_img, footprint)
    plot_comparison(bin_img, opened, 'opening')


def close_img(im_org):
    footprint = disk(1)
# Check the size and shape of the structuring element
    print(footprint)
    bin_img = otsu(gray(im_org))
    closed = closing(bin_img, footprint)
    plot_comparison(bin_img, closed, 'closing')


def ots_comp(im_org):
    plot_comparison(im_org, otsu(gray(im_org)), "Otsu thresholding")


# Ex 6, computing outline
def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    bin_img = otsu(gray(im_org))
    bin_img = np.invert(bin_img)
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline


def compute_outline_bin(bin_img):
    """
    Computes the outline of a binary image
    """

    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline

# Ex 7, another way of computing outline. Looks wack due to small opening, removes very small objects. THen we have a very big closing, which means that all the remains get enlarged heavily. This is why we have overlappign circles on the edges of the lego


def compute_outline2(im_org):
    bin_img = otsu(gray(im_org))
    footprint = disk(1)
    big_foot = disk(15)
    dilated = opening(bin_img, footprint)
    closed = closing(dilated, big_foot)
    outline = np.logical_xor(closed, bin_img)
    return outline

# ex 8, morphology with multiple objects.


def legos(im_org):
    bin_img = otsu(gray(im_org))
    outline_img = compute_outline(im_org)
    # plot_comparison(im_org, bin_img, "binary")
    plot_comparison(bin_img, outline_img, "binayr vs outline")


# Ex 9, Closing.
def compute_outline3(bin_img):
    """
    Computes the outline of a binary image
    """
    # Have to invert the picture according to our TA. The relevant stuff should be white, NOT black.
    bin_img = otsu(gray(im_org))

    bin_img = np.invert(bin_img)

    footprint = disk(5)
    dilated = closing(bin_img, footprint)
    outline = np.logical_xor(bin_img, dilated)
    return outline


def lego2(im_org):
    bin_img = otsu(gray(im_org))
    bin_img = np.invert(bin_img)
    outline_img = compute_outline3(im_org)
    # Have to invert the picture according to our TA. The relevant stuff should be white, NOT black.

    # plot_comparison(im_org, bin_img, "binary")
    plot_comparison(bin_img, outline_img, "binary vs outline")


# Ex 11

def ex11(im_org):
    bin_img = otsu(gray(im_org))
    bin_img = np.invert(bin_img)
    footprint = disk(25)
    # Closing removes the inner "dutter" from the legos
    outline = closing(bin_img, footprint)
    # Erosion is to make sure they no longer touch. Does not work.
    outline = erosion(outline, disk(64))
    outline = compute_outline_bin(outline)
    # Erosion to remove images from each other
    plot_comparison(bin_img, outline, "what")


# Directory containing data and images
in_dir = "exercises/ex4b-ImageMorphology/data/"

# X-ray image
# im_name = "DTUSign1.jpg"
im_name = "Lego_9.png"
im_org = io.imread(in_dir + im_name)
# open_img(im_org)
# show(compute_outline(im_org))
# show(compute_outline2(im_org))
# lego2(im_org)
ex11(im_org)
