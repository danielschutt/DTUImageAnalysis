import imagecodecs
import itertools
from skimage.color import label2rgb
from skimage import measure
from skimage import segmentation
import math
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
from skimage.morphology import erosion, dilation, opening, closing, binary_closing, binary_opening
from skimage.morphology import disk
from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
##### HELPERS ##########


def show(input_img, min=0, max=255):
    io.imshow(input_img, vmin=min, vmax=max, cmap='gray')
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


def myHist(data, bins):
    plt.hist(data, bins=bins)
    io.show()


###### END H ELPERS #####

# ex 2


def getOrganHelper(loc):
    spleen_roi = io.imread(loc)
    # convert to boolean image
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]
    return spleen_values


# ex 4, gaussian distribution fit. Bone values are not gaussian, rest are.
def gss(data, type):
    n, bins, patches = plt.hist(data, 60, density=1)
    pdf_spleen = norm.pdf(bins, np.mean(data), np.std(data))
    plt.plot(bins, pdf_spleen)
    plt.xlabel('Hounsfield unit')
    plt.ylabel('Frequency')
    plt.title(type + ' values in CT scan')
    plt.show()

    # Displaying multiple at once


def gss_multiple(data1, data2, label1="Data1", label2="data2", tit="Comparison of gaussians"):
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, np.mean(data1), np.std(data1))
    pdf_bone = norm.pdf(hu_range, np.mean(data2), np.std(data2))
    plt.plot(hu_range, pdf_spleen, 'r-', label=label1)
    plt.plot(hu_range, pdf_bone, 'b', label=label2)
    plt.title(tit)
    plt.legend()
    plt.show()
    # Spleen and liver are VERY close, spleen and kidney are also close. See below code, change first variable for ease of use

# gss_multiple(kidney_values, spleen_values, "Spleen","kidney", "Spleen kidney comparison")
# gss_multiple(kidney_values, fat_values, "Spleen","fat", "Spleen fat comparison")
# gss_multiple(kidney_values, liver_values, "Spleen","liver", "Spleen liver comparison")

# gss_multiple(kidney_values, bone_values, "Spleen","bone", "bone kidney comparison")
# gss_multiple(kidney_values, kidney_values, "Spleen","kidney", "Spleen kidney comparison")
# ex 7, 8 minimum distance classification.


# This part displays the classification. It is very good at finding fat and soft tissue, it does however suck at finding bone.
# label_img = fat_img + 2 * soft_img + 3 * bone_img
# image_label_overlay = label2rgb(label_img)
# plot_comparison(img, image_label_overlay, 'Classification result')

# PARAMETRIC PIXEL CLASSIFICATION

# exercise 9, find intersections of gaussians for thresholding.

def ex9(soft, fat, test_value, show):
    if norm.pdf(test_value, np.mean(soft), np.std(soft)) > norm.pdf(test_value, np.mean(fat), np.std(fat)):
        print(f"For value {test_value} the class is #1")
    else:
        print(f"For value {test_value} the class is #2")
    if show:
        gss_multiple(soft, fat, "soft", "fat")

# ex 11, find single object based on their values. BLOB analysis and thresholding.

# This is actually quite good for finding the spleen, it is quite clearly separated from the remaining blobs. However there are still a ton of other blobs.


def ex11(spleen_values, img):
    # First choose spleen value. This is an awful estimation. Do not use this.
    t_1 = np.mean(spleen_values)-(np.std(spleen_values)/20)
    t_2 = np.mean(spleen_values)+(np.std(spleen_values)/20)
    spleen_estimate = (img > t_1) & (img < t_2)
    spleen_label_colour = color.label2rgb(spleen_estimate)
    footprint = disk(8)
    closed = binary_closing(spleen_estimate, footprint)

    footprint = disk(5)
    opened = binary_opening(closed, footprint)
    label_img = measure.label(opened)
    image_label_overlay = label2rgb(label_img)
    # show(image_label_overlay)
    stricter(label_img, image_label_overlay)

# ex 13,14, blob analysis based off of area and perimeter


def stricter(label_img, comp_pic):
    # min and max found via experimentation. Print the regions, then make educated guesses
    min_area = 3000
    max_area = 5000
    min_perimeter = 250
    max_perimeter = 400
    region_props = measure.regionprops(label_img)

    # Create a copy of the label_img
    label_img_filter = label_img.copy()
    for region in region_props:
        print(region.perimeter)
        # Find the areas that do not fit our criteria
        if (region.area > max_area or region.area < min_area) or (region.perimeter > max_perimeter or region.perimeter < min_perimeter):

            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    plot_comparison(comp_pic, i_area, 'Found spleen based on area')

# Ex 15, spleen finder


def spleen_finder(img):
    getOrganHelper(in_dir + 'SpleenROI.png')
    ex11(spleen_values, img)


# Object segmentation, spleen finder from here.
# Exercises from here!
in_dir = "exercises/ex6-PixelClassificationAndObjectSegmentation/data/"
# Original training picture
# ct = dicom.read_file(in_dir + 'validation1.dcm')
# Validation pictures
# ct = dicom.read_file(in_dir + 'validation1.dcm')
# ct = dicom.read_file(in_dir + 'validation2.dcm')
ct = dicom.read_file(in_dir + 'validation3.dcm')
img = ct.pixel_array
print(img.shape)
print(img.dtype)


spleen_values = getOrganHelper(in_dir + 'SpleenROI.png')
kidney_values = getOrganHelper(in_dir + 'KidneyROI.png')
fat_values = getOrganHelper(in_dir + 'FatROI.png')
liver_values = getOrganHelper(in_dir + 'LiverROI.png')
bone_values = getOrganHelper(in_dir + 'BoneROI.png')

# minimum distance classification from here

# First combine soft tissue values
soft_tissue = itertools.chain(spleen_values, kidney_values, liver_values)
soft_tissue = list(soft_tissue)
avg_soft = np.mean(soft_tissue)
avg_bone = np.mean(bone_values)
avg_fat = np.mean(fat_values)
# Calculate thresholds by minimum distance classifier
t_fat_soft = (avg_soft+avg_fat)/2
t_soft_bone = (avg_soft+avg_bone)/2

t_background = -200
# Calculate images
fat_img = (img > t_background) & (img <= t_fat_soft)
soft_img = (img > t_fat_soft) & (img <= t_soft_bone)
bone_img = (img > t_soft_bone)


# PArametric classification from here.
# Threshold around -44 between soft tissue and fat values. Found using ex9 code, then guessing values. Use the plot for more accurate comparison
# ex9(soft_tissue, fat_values, -44)
t_fat_soft = -44
# Other threshold is 140 between bone and soft tissue
ex9(soft_tissue, bone_values, 140, False)
t_soft_bone = 140
# show(img)


# OBJECT SEGMENTATION - SPLEEN FINDER FROM HERE
spleen_finder(img)
