from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import skimage as sk
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom


def grey(im_org, min, max):
    io.imshow(im_org, vmin=min, vmax=max)
    plt.title('Metacarpal image (with gray level scaling)')
    io.show()
# Exercise 8, displaying histogram


def hist(im):
    plt.hist(im.ravel(), bins=256)

    plt.title('Image histogram')
    io.show()

# Exercise 9, using the histogram function to find the most common intensity


def hist_intensity(im):
    h = plt.hist(im.ravel(), bins=256)
    bin_no = 100
    print("Maximum indice is " +
          str(np.argmax(h[0])) + " Which has " + str(np.max(h[0])) + " values")
    plt.title('Image histogram')
    io.show()

# Exercise 10, pixel values


def RC(im, r, c):

    im_val = im[r, c]
    print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# Ex 11


def slice(im_org):
    im_org[:30] = 0
    io.imshow(im_org)
    io.show()

# Ex 12, mask where everything that doesn't comply with mask is black


def mask(im_org):
    mask = im_org > 150
    io.imshow(mask)
    io.show()

# Ex 13, masking applied differently, where everything else is the old color


def mask_v2(im_org):
    mask = im_org > 150
    im_org[mask] = 255
    io.imshow(im_org)
    io.show()


# Ex 16, coloring picture with slice


def slice_v2(im_org):

    im_org[:int(im_org.shape[0]/2)] = [0, 255, 0]
    io.imshow(im_org)
    io.show()

# ex 18, resize


def resize(im_org):
    # Normalizes colors
    im_org = rescale(im_org, 0.25, anti_aliasing=True,
                     channel_axis=2)
    print(im_org.shape)
    print(im_org)
    io.imshow(im_org)
    io.show()

# ex 19, resize v2, column = 400


def resize_v2(im_org):
    X = im_org.shape[1] // 400

    im_org = sk.transform.resize(im_org, (im_org.shape[0] // 4,
                                          im_org.shape[1] // X),
                                 anti_aliasing=True)
    print(im_org.shape)
    print(im_org)
    io.imshow(im_org)
    io.show()

# Ex 20, compare histograms


def hist_v3(im1, im2):
    hist(im1)
    hist(im2)

# ex 22, show DTU


def display(im):
    io.imshow(im)
    io.show()
# Ex 23


def displayDTU(im, val):
    r_comp = im[:, :, val]
    io.imshow(r_comp)
    plt.title('DTU sign image (Red)')
    io.show()


def displayALL(im):
    display(im)
    displayDTU(im, 0)
    displayDTU(im, 1)
    displayDTU(im, 2)

# ex 25, save


def BlockAndSave(im):
    im[500:1000, 800:1500, :] = 0
    display(im)
    io.imsave(in_dir + "Blocked.png", im)

# ex 27, change color


def changeBones(location):
    im_org = io.imread(location, as_gray=True)
    im_color = color.gray2rgb(im_org)
    im_byte = img_as_ubyte(im_color)

    mask = im_org > 140
    im_byte[mask] = [0, 0, 255]

    display(im_byte)

# ex 28, advanced image analysis


def Adv(im_org):
    p = profile_line(im_org, (342, 77), (320, 160))
    plt.plot(p)
    plt.ylabel('Intensity')
    plt.xlabel('Distance along line')
    plt.show()

# ex 28:


def HeightAsGray():
    in_dir = "exercises/ex1-IntroductionToImageAnalysis/data/"
    im_name = "road.png"
    im_org = io.imread(in_dir + im_name)
    im_gray = color.rgb2gray(im_org)
    ll = 200
    im_crop = im_gray[40:40 + ll, 150:150 + ll]
    xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                           linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Directory containing data and images
in_dir = "exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
# im_name = "DTUSign1.jpg"
im_name = "metacarpals.png"
# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.

im_org = io.imread(in_dir + im_name)
im2 = io.imread(in_dir + "dark.jpg")

print(im_org.shape)
print(im_org.dtype)
print(im_org)


# You can see original grey picture by having it without cmap=jet.
# io.imshow(im_org, cmap="hot")
# plt.title('Metacarpal image (with colormap)')
# io.show()

# Exercise 7, calculating lowest and highest values of colors in the image.
min = np.min(im_org)
max = np.max(im_org)
# grey(im_org, min, max)


HeightAsGray()
