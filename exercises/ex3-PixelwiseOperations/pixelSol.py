from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu
import skimage as sk
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom


def display(im):
    io.imshow(im)
    io.show()
# Exercise 1, displaying histogram


def hist(im):
    plt.hist(im.ravel(), bins=256)

    plt.title('Image histogram')
    io.show()

# Exercise 2-4, converting to float.


def conversion(im_org):
    im_float = img_as_float(im_org)
    img_BYTE = img_as_ubyte(im_float)
    print(np.min(im_org))
    print(np.max(im_org))
    print(np.min(im_float))
    print(np.max(im_float))
    print(np.min(img_BYTE))
    print(np.max(img_BYTE))

# Exercise 5, histogram where Vmax and Vmin are desired max andm in.


def histogram_stretch(img_in):
    """
    Stretches the histogram of an image
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0

    # Do something here
    min = np.min(img_float)
    max = np.max(img_float)
    img_out = ((max_desired-min_desired)/(max_val-min_val)) * \
        (img_float-min_val)+min_desired
    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)

# exercise 6, Stretches and compares two pictures


def compareStretch(im_org):
    display(im_org)
    im_org = histogram_stretch(im_org)
    display(im_org)


# Exercise 7, non-linear pixel gamma mapping

def GammaPower(im_org, gamma):

    im_float = img_as_float(im_org)
    im_gam = np.power(im_float, gamma)
    return im_gam

# Exercise 8 displaying gamma difference


def gammaDif(im_org):
    im_gamma = GammaPower(im_org, 2.0)
    display(im_org)
    display(im_gamma)

# Exercise 9, thresholding an image into foreground and background


def threshold_image(img_in, thres):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    im_float = img_as_float(img_in)
    im_float[im_float > thres] = 1.0
    im_float[im_float < thres] = 0.0
    return img_as_ubyte(im_float)

# Exercise 10, thresholding in action. Impossible to separate neck bones from rest of pic.


def thresh(im_org, thresh):
    im_thresh = threshold_image(im_org, thresh)
    return im_thresh


# Exericse 11, comparing otsu threshold to my own
def otsu(im_org):
    im_float = img_as_float(im_org)
    otsu_thresh = threshold_otsu(im_float)
    display(thresh(im_float, otsu_thresh))

# Exercise 12-14 color thresholding/isolation. Isolates color in the signs which are red and blue


def color_thresh(im_org):
    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    segm_blue = (r_comp < 10) & (g_comp > 85) & (g_comp < 105) & \
        (b_comp > 180) & (b_comp < 200)

    segm_red = (r_comp > 150) & (r_comp < 190) & (g_comp > 40) & (
        g_comp < 70) & (b_comp < 70) & (b_comp > 40)
    im_org[~segm_blue] = [0, 0, 0]
    im_org[~segm_red] = [0, 0, 0]

    display(im_org)


# Color thresholding in HSV space.
def color_HSV(im_org):
    hsv_img = color.rgb2hsv(im_org)
    hue_img = hsv_img[:, :, 0]
    value_img = hsv_img[:, :, 2]
    hist(hue_img)
    segm_red = (hue_img > 0.8)
    hsv_img[~segm_red] = [0, 0, 0]

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
    ax0.imshow(im_org)
    ax0.set_title("RGB image")
    ax0.axis('off')
    ax1.imshow(hue_img, cmap='hsv')
    ax1.set_title("Hue channel")
    ax1.axis('off')
    ax2.imshow(value_img)
    ax2.set_title("Value channel")
    ax2.axis('off')

    fig.tight_layout()
    io.show()

    # Directory containing data and images
in_dir = "exercises/ex3-PixelwiseOperations/data/"


im_name = "DTUSigns2.jpg"

im_org = io.imread(in_dir + im_name)

color_HSV(im_org)
