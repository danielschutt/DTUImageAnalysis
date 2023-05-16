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
from skimage import filters
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import imagecodecs


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
# exercise 3, closing and opening pictures. Requires a binary image!


def close_img(im_org, diskSize):
    footprint = disk(diskSize)
    return closing(im_org, footprint)


def open_img(im_org, diskSize):
    footprint = disk(diskSize)
    return opening(im_org, footprint)

# ex 4, FIND BLOBS


def CPA(img_open):
    label_img = measure.label(img_open)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")


# ex 5, display blobs
def reRGB(img_open):
    label_img = label2rgb(img_open)
    return label_img

# ex 6, compute  blob features


def blob_feat(label_img):

    region_props = measure.regionprops(label_img.astype(int))
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()


# Ex 9, another way of computing blobs and showing each blob with different color

def ex9(img_c_b):
    label_img = measure.label(img_c_b)
    image_label_overlay = label2rgb(label_img)
    return image_label_overlay


# ex 10, object features
def ex10(img_c_b):
    label_img = measure.label(img_c_b)
    region_props = measure.regionprops(label_img)
    print(region_props[0].area)
    return label_img

# Ex 11, calculate the area of blobs and display them on a histogram. Here we're interested in removing big blobs.


def ex11(img_c_b):
    label_img = measure.label(img_c_b)
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas.ravel(), bins=256, range=(60, 100))
    io.show()

    return label_img


# ex 12, removing too large blobs.

def ex12(img_c_b):
    label_img = measure.label(img_c_b)
    region_props = measure.regionprops(label_img)
    # Numbers found by looking at histogram.
    min_area = 50
    max_area = 80

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    plot_comparison(img_small, i_area, 'Found nuclei based on area')

# ex 12, compare area to perimeter


def ex12b(img_c_b):
    label_img = measure.label(img_c_b)

    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    peris = np.array([prop.perimeter for prop in region_props])
    plt.hist(areas.ravel(), bins=256, range=(60, 100))
    io.show()
    plt.hist(peris.ravel(), bins=256, range=(1, 100))
    io.show()
    return label_img

# Ex 13, calculate circularity


def ex13(img_c_b):
    label_img = measure.label(img_c_b)
    region_props = measure.regionprops(label_img)
    print(f"Number of labels: { label_img.max()}")
    areas = np.array([prop.area for prop in region_props])
    peris = np.array([prop.perimeter for prop in region_props])

    circularity = (4*math.pi*areas)/(np.power(peris, 2))

    plt.hist(circularity.ravel(), bins=256, range=(0.0, 1.0))
    io.show()
    return [label_img, region_props]


# Display only nuclei with a certain circularity
def ex13b(label_img, region_props):
    areas = np.array([prop.area for prop in region_props])
    peris = np.array([prop.perimeter for prop in region_props])

    circularity = (4*math.pi*areas)/(np.power(peris, 2))

    min_circ = 0.7
    circ_mask = circularity > min_circ
    print(circ_mask)

    label_img_filter = label_img
    for i in range(0, len(circ_mask)):
        if not circ_mask[i]:
            for cords in region_props[i].coords:
                label_img_filter[cords[0], cords[1]] = 0

    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    plot_comparison(img_small, i_area, 'Found nuclei based on circularity')


# Code begin here
# Directory containing data and images
in_dir = "exercises/ex5-BLOBAnalysis/data/"

# X-ray image
# im_name = "DTUSign1.jpg"
im_name = "Lego_4_small.png"
im_org = io.imread(in_dir + im_name)

thresholded = np.invert(otsu(gray(im_org)))


# cleared = clear_border(thresholded)
# closed = close_img(cleared, 5)
# opened = open_img(closed, 5)
# CPA(opened)
# rgbPic = reRGB(opened)
# blob_feat(rgbPic)
# plot_comparison(im_org, rgbPic, "Thresholded")


# Part 2 of the exercise CELL COUNTING
in_dir = "exercises/ex5-BLOBAnalysis/data/"
img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small)
# avoid bin with value 0 due to the very large number of background pixels


grayPic = otsu(img_gray)
cleared = clear_border(grayPic)
blobby = ex13(cleared)
ex13b(blobby[0], blobby[1])
# plot_comparison(img_gray, blobby, "plt")
