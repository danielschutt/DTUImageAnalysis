from scipy.ndimage import correlate
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.filters import median
import skimage as sk

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
import matplotlib.pyplot as plt
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
from skimage.transform import warp
from skimage.transform import swirl


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


def show_comparison(original, transformed, transformed_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(transformed)
    ax2.set_title(transformed_name)
    ax2.axis('off')
    io.show()


def gray(input_img):
    new_img = color.rgb2gray(input_img)
    return new_img


# Exercises from here
def myRotate(im_org, rotation_angle):
    # change 1.0 to 0.0 for black broder
    return rotate(im_org, rotation_angle, resize=True, mode="constant", cval=1.0)


# ex 7, transformed rotation. rotation in degrees, not rad
def transformer(im_org, rotation_angle):
    # angle in radians - counter clockwise
    rotation_angle = 10.0 * math.pi / 180.
    trans = [0, 0]
    tform = EuclideanTransform(rotation=rotation_angle, translation=trans)
    # You can get the inverse by using tform.inverse
    return warp(im_org, tform)

# ex 9, similarity transform.


def simTrans(im_org, rotation_angle=15, trans=[40, 30], scale=0.6):
    # angle in radians - counter clockwise
    rotation_angle = 10.0 * math.pi / 180.

    tform = SimilarityTransform(scale=scale, rotation=rotation_angle,
                                translation=trans)
    # You can get the inverse by using tform.inverse
    return warp(im_org, tform)


def swirlTrans(im_org):
    str = 10
    rad = 300
    return swirl(im_org, strength=str, radius=rad)

# ex 11, displays two images blended together


def blender(src_img, dst_img):
    blend = 0.5 * img_as_float(src_img) + 0.5 * img_as_float(dst_img)
    io.imshow(blend)
    io.show()


def displayLandmarks(src_org, src):
    plt.imshow(src_img)
    plt.plot(src[:, 0], src[:, 1], '.r', markersize=12)
    plt.show()
# ex 13, Calculates how well two sets of landmarks are to each other


def calcAlignment(src, dst):
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    return f

# ex 15, moving from source landmarks to destination landarks


def moveEst(src_img, src, dst):
    tform = EuclideanTransform()
    tform.estimate(src, dst)
    src_transform = matrix_transform(src, tform.params)
    return warp(src_img, tform.inverse)


# Regular code from here
# Directory containing data and images
in_dir = "exercises/Ex7-GeometricTransformationsAndRegistration/data/"


im_name1 = "Hand1.jpg"
im_name2 = "Hand2.jpg"
src_img = io.imread(in_dir + im_name1)
# Landmarks of source
src = np.array([[588, 274], [328, 179], [134, 398], [260, 525], [613, 448]])


dst_img = io.imread(in_dir + im_name2) 
# Destination alndmarks found manually
dst = np.array([[636, 310], [377, 167], [200, 276], [279, 439], [597, 451]])
print(calcAlignment(src, dst))

# Final changes, and how to move the pictures to align.
changed_pic = moveEst(src_img, src, dst)
blender(dst_img, changed_pic)
show_comparison(dst_img, changed_pic, "attempt at transform")
