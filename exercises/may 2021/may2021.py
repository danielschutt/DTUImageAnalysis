from scipy.ndimage import correlate
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
from skimage.filters import median
import skimage as sk
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
from LDA import LDA
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
import pydicom as dicom
from skimage.filters import gaussian
from skimage.filters import prewitt_h
from skimage.filters import prewitt_v
from skimage.filters import prewitt
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import threshold_otsu
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
import numpy as np
from scipy.stats import norm
import math
from skimage.transform import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
from skimage.transform import warp
from skimage.transform import swirl
import numpy as np
from skimage.segmentation import clear_border
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize
from skimage.color import rgb2gray
from mat4py import loadmat
from scipy.spatial import distance
#HELPER FUNCTIONS
def mean_filt(input_img, filterSize):
    size = filterSize
    # Two dimensional filter filled with 1
    weights = np.ones([size, size])
    # Normalize weights
    weights = weights / np.sum(weights)
    res_img = correlate(input_img, weights)
    return res_img
def GammaPower(im_org, gamma):

    im_float = img_as_float(im_org)
    im_gam = np.power(im_float, gamma)
    return im_gam
def histogram_stretch(img_in,lower,upper):
    """
    Stretches the histogram of an image
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    #min_desired = 0.0
    #max_desired = 1.0
    min_desired = lower/255
    max_desired = upper/255

    # Do something here
    min = np.min(img_float)
    max = np.max(img_float)
    img_out = ((max_desired-min_desired)/(max_val-min_val)) * \
        (img_float-min_val)+min_desired
    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)


def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice    
def gray(input_img):
    new_img = color.rgb2gray(input_img)
    return new_img
def median_filt(input_img, filterSize):
    size = filterSize
    footprint = np.ones([size, size])
    med_img = median(input_img, footprint)
    return med_img
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
def dice(pred, true, k = 1):
    true = true > 0
    return 1 - distance.dice(pred.ravel(), true.ravel())

def simTrans(im_org, rotation_angle=0, trans=[40, 30], scale=1):
    # angle in radians - counter clockwise
    rotation_angle = rotation_angle * math.pi / 180.

    tform = SimilarityTransform(scale=scale, rotation=rotation_angle,
                                translation=trans)
    # You can get the inverse by using tform.inverse
    return warp(im_org, tform)
def show(input_img):
    io.imshow(input_img)
    io.show()

def plot_comparison(original, filtered, filter_name="undefined"):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    io.show()
    
def threshold_image(img_in, thres, convert=True):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    if(convert):
        img_in = img_as_ubyte(img_in)
    
    img_in[img_in > thres] = 255
    img_in[img_in < thres] = 0
    return img_in
def threshold_image_reverse_border(img_in, thres,convert=True):
    """
    Apply a threshold in an image and return the resulting image
    :param img_in: Input image
    :param thres: The treshold value in the range [0, 255]
    :return: Resulting image (unsigned byte) where background is 0 and foreground is 255
    """
    if(convert):
        img_in = img_as_ubyte(img_in)
    
    img_in[img_in < thres] = 255
    img_in[img_in > thres] = 0
    return img_in

def thresh(im_org, thresh):
    im_thresh = threshold_image(im_org, thresh)
    return im_thresh
#Requires binary img
def close_img(im_org, diskSize):
    footprint = disk(diskSize)
    return closing(im_org, footprint)
def open_img(im_org, diskSize):
    footprint = disk(diskSize)
    return opening(im_org, footprint)

def moveEst(src_img, src, dst):
    tform = EuclideanTransform()
    tform.estimate(src, dst)
    src_transform = matrix_transform(src, tform.params)
    return warp(src_img, tform.inverse)

#EXERCISES -------------------------------------------------------------------------------------------------------
def iris():
    in_dir = "exercises/may 2021/data/"
    txt_name = "irisdata.txt"
    car_data = np.loadtxt(in_dir + txt_name, comments="%")
    x = car_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")


    mn = np.mean(x, axis=0)
    data = x - mn
    #Tror måske ikke det er meningen at fuck med standard deviation?
    #data = data / data.std(axis=0)
    c_x = np.cov(data.T)
    

    values, vectors = np.linalg.eig(c_x)
  

    

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    #plt.show()

  
    answer = v_norm[0] + v_norm[1]
    print(f"Answer: Variance explained by the first two PC: {answer:.2f}")

    
    
   # Project data
    pc_proj = vectors.T.dot(data.T)
    pc_proj = vectors.T @ data.T
    #Data = M
    #print(data)
    print(vectors.T)

def sky():
    in_dir = "exercises/may 2021/data/"
    txt_name = "sky_gray.png"
    im_org = io.imread(in_dir+txt_name)
    
    big_stretch = histogram_stretch(im_org,10,200)
    #plot_comparison(im_org,big_stretch)
    #print(np.mean(im_org))
    #print(np.mean(big_stretch))


    #Going for the filtering question
    gam = GammaPower(im_org,1.21)
    med = median_filt(gam,5)
    med = img_as_ubyte(med)
    print(med[40,50])

    ####SECOND ANSWER

    txt_name = "sky.png"
    im_org = io.imread(in_dir+txt_name)
    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    
    segm = (r_comp < 100) & (g_comp > 85) &  (b_comp > 150) 
    
    im_org[segm] = 1.0
    im_org[~segm] = 0
    test = im_org


    im_org = gray(im_org)
    dis = disk(5)
    
    im_org = erosion(im_org,dis)
    #plot_comparison(test,im_org)
      #EXAM ANSWER
    #print("ANSWER")
    #print(np.sum(im_org > 0))

def flower():
    in_dir = "exercises/may 2021/data/"
    txt_name = "flower.png"
    im_org = io.imread(in_dir+txt_name)
    im_org = rgb2hsv(im_org)

    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
        
    segm = (r_comp < 0.25) & (g_comp > 0.8) &  (b_comp > 0.8) 

    im_org = np.resize(im_org,(600,800))
    im_org[segm] = 255
    im_org[~segm] = 0
    
    #im_org = gray(im_org)
    
    opened = open_img(im_org, 5)

    #plot_comparison(im_org,opened)
    print(np.sum(opened > 0))


    #FLOWERWALL
    in_dir = "exercises/may 2021/data/"
    txt_name = "flowerwall.png"
    im_org = io.imread(in_dir+txt_name)

    im = mean_filt(im_org,15)
    #EXAM ANSWER
    print("EXAM ANSWER")
    print(im[5,50])

def car():
    in_dir = "exercises/may 2021/data/"
    car1 = "car1.jpg"
    car2 = "car2.jpg"
    car3 = "car3.jpg"
    car4 = "car4.jpg"
    car5 = "car5.jpg"
    

    car1 = io.imread(in_dir + car1)
    car2 = io.imread(in_dir + car2)
    car3 = io.imread(in_dir + car3)
    car4 = io.imread(in_dir + car4)
    car5 = io.imread(in_dir + car5)
    blend = 1/5 * img_as_float(car1) + 1/5 * img_as_float(car2) + 1/5 * img_as_float(car3) + 1/5 * img_as_float(car4) + 1/5 * img_as_float(car5) 
    
    mn = np.mean(blend)
    data = blend - mn
    #data = data / data.std(axis=0)
    c_x = np.cov(data.T)

    print(f"First answer at top of matrix {data[0][0]:.2f}")

    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    answer = v_norm[0] + v_norm[1]
    print(f"Answer: Variance explained by the first two PC: {answer:.2f}")

#TEST AREA IGNORE PLS ---------------------------------------------------
flower()