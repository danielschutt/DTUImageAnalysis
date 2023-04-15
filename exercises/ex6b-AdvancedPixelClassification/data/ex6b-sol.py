import numpy as np
from LDA import LDA
import sys
import matplotlib.pyplot as plt
import scipy.io as sio
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


def show(input_img, min=0, max=1):
    io.imshow(input_img, vmin=min, vmax=max)
    io.show()


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


def myHist(data, bins):
    plt.hist(data, bins=bins)
    io.show()


in_dir = "exercises/ex6b-AdvancedPixelClassification/data/"
in_file = 'ex6_ImagData2Load.mat'
data = sio.loadmat(in_dir + in_file)
ImgT1 = data['ImgT1']
ImgT2 = data['ImgT2']
ROI_GM = data['ROI_GM']
ROI_WM = data['ROI_WM']
C1 = ROI_GM
C2 = ROI_WM
training_vector = np.array(ImgT1 + ImgT2)
training_vector = training_vector+2
target_vector = np.array(ROI_GM + ROI_WM)
print(str(len(ImgT1)) + " " + str(len(ImgT2)))
print(str(len(ROI_GM)) + " " + str(len(ROI_WM)))
print(training_vector)
# W = LDA(training_vector, target_vector)
# show(ROI_GM)
