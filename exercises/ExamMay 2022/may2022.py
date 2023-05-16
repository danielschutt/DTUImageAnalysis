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
    rotation_angle = 10.0 * math.pi / 180.

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
#EXERCISES FROM HERE

#climbing wall
def ex1():
    in_dir = "exercises/ExamMay 2022/data/Climbing/"
    im_name = "ClimbingWall.png"
    im_org = io.imread(in_dir + im_name)
    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    
    segm = (r_comp < 60) & (g_comp < 200) &  (b_comp < 100) 
    
    im_org[segm] = 255
    im_org[~segm] = 0
    im_org = rgb2gray(im_org)
    
    change = close_img(im_org,3)
    change = clear_border(change)

    

    label_img = measure.label(change, connectivity =2)
    region_props = measure.regionprops(label_img)
    print(len(region_props))
    #plot_comparison(im_org,change, "After closing")
    counter = 0
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > 300 and region.perimeter < 500:
            counter = counter +1

    print(counter)




def ex2():
    fixed = loadmat('exercises/ExamMay 2022/data/Donald/donaldfixedPoints.mat')['fixedPoints']
    moving = loadmat('exercises/ExamMay 2022/data/Donald/donaldmovingPoints.mat')['movingPoints']

    in_dir = "exercises/ExamMay 2022/data/Donald/"
    im_name = "donald_2.png"
    
    im_org = io.imread(in_dir + im_name)
    im_org = img_as_ubyte(im_org)

    im2 = moveEst(im_org,moving,fixed)
    
    im2 = img_as_ubyte(im2)
    
    print(im2[299,299])
    #Part 2, calculate center of mass
    mf= np.mean(fixed)
    mm = np.mean(moving)
    ex = mf-mm
    print(np.sqrt(ex*ex))
    dist = np.linalg.norm(np.mean(fixed)-np.mean(moving))
    print(dist)



def ex3():
    in_dir = "exercises/ExamMay 2022/data/ImagePCA/"
    orchid1 = "orchid001.png"
    orchid2 = "orchid002.png"
    orchid3 = "orchid003.png"
    orchid4 = "orchid004.png"
    orchid5 = "orchid005.png"
    orchid6 = "orchid006.png"

    orchid1 = io.imread(in_dir + orchid1)
    orchid2 = io.imread(in_dir + orchid2)
    orchid3 = io.imread(in_dir + orchid3)
    orchid4 = io.imread(in_dir + orchid4)
    orchid5 = io.imread(in_dir + orchid5)
    orchid6 = io.imread(in_dir + orchid6)

    blend = 1/6 * img_as_float(orchid1) + 1/6 * img_as_float(orchid2) + 1/6 * img_as_float(orchid3) + 1/6 * img_as_float(orchid4) + 1/6 * img_as_float(orchid5) + 1/6 * img_as_float(orchid6)
    
    im = thresh(blend,150)
    
    show(im)

def ex4():
    in_dir = "exercises/ExamMay 2022/data/PCAData/"
    txt_name = "pizza.txt"
    car_data = np.loadtxt(in_dir + txt_name, comments="%")

    x = car_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")

    # plt.figure()
    # Transform the data into a Pandas dataframe
    # d = pd.DataFrame(x)
    # sns.pairplot(d)
    # plt.show()

    mn = np.mean(x, axis=0)
    data = x - mn
    data = data / data.std(axis=0)
    c_x = np.cov(data.T)



    values, vectors = np.linalg.eig(c_x)

    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.show()

    print("All vnorms")
    print(v_norm)
    answer = v_norm[0] + v_norm[1] + v_norm[2]  
    print(f"Answer: Variance explained by the first two PC: {answer:.2f}")


def ex5():
    in_dir = "exercises/ExamMay 2022/data/DICOM/"
    ct = dicom.read_file(in_dir + '1-131.dcm')
    img = ct.pixel_array

    liver_roi = io.imread(in_dir + "LiverROI.png")
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    # liver_mean = np.average(liver_values)
    # liver_std = np.std(liver_values)
    (mu_liver, std_liver) = norm.fit(liver_values)
    

    bone_roi = io.imread(in_dir + "BoneROI.png")
    bone_mask = bone_roi > 0
    bone_values = img[bone_mask]
    # liver_mean = np.average(liver_values)
    # liver_std = np.std(liver_values)
    (mu_bone, std_bone) = norm.fit(bone_values)

    spleen_roi = io.imread(in_dir + "SpleenROI.png")
    spleen_mask = spleen_roi > 0
    spleen_values = img[spleen_mask]
    (mu_spleen, std_spleen) = norm.fit(spleen_values)

    t_liver_kidney = (mu_liver + mu_spleen) / 2
    t_kidney_aorta = (mu_liver + mu_bone) / 2
    print(t_liver_kidney)
    print(t_kidney_aorta)
    #NEXT EXERCISE IN HERE 

def ex6():
    in_dir = "exercises/ExamMay 2022/data/BikeImage/"
    im_name = "bikes.png"
    im_org = io.imread(in_dir + im_name)
    ground_truth = io.imread(in_dir + "boxROI.png")
    r_comp = im_org[:, :, 0]
    g_comp = im_org[:, :, 1]
    b_comp = im_org[:, :, 2]
    
    segm = (r_comp < 100) & (g_comp > 200) &  (b_comp > 100) 
    
    im_org[segm] = 255
    im_org[~segm] = 0
    im_org = rgb2gray(im_org)
    
    change = close_img(im_org,8)
    show(change)
    print(dice(change,ground_truth))


def ex7():
    grass = [68,65,67]
    road =  [70,80,75]
    sky = [77,92,89]

    m1  =np.mean(grass)
    m2  =np.mean(road)
    m3  =np.mean(sky)

    t1 = (m1+m2)/2
    t2 = (m2+m3)/2
    print(t1)
    print(t2)


def ex8():
    in_dir = "exercises/ExamMay 2022/data/DICOM/"
    ct = dicom.read_file(in_dir + '1-131.dcm')
    img = ct.pixel_array

    t1=85
    t2=400
    
    img[img > t2] = 255
    img[img <t1] = 0
    
    img_2d = img.astype(float)


    img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
    img = np.uint8(img_2d_scaled)
    
    closed = close_img(img,5)
    closed = open_img(closed,3)
    plot_comparison(img,closed)
    label_img = measure.label(closed, connectivity =2)
    region_props = measure.regionprops(label_img)
    counter = 0
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area < 4000 or region.area > 1000:
            counter = counter +1
            # set the pixels in the invalid areas to background
    print(counter)

def ex9():
    in_dir = "exercises/ExamMay 2022/data/Bird/"
    im_name = "bird.png"
    im_org = io.imread(in_dir + im_name)
    im_org = cv2.cvtColor(im_org, cv2.COLOR_BGR2HSV)
    
    
    im_org = im_org[:,:,1]
    print(im_org)
    im_org = prewitt_filt(im_org, "v")
    show(im_org)


def ex10():
    in_dir = "exercises/ExamMay 2022/data/Water/"
    im_name = "water_gray.png"
    im_org = io.imread(in_dir + im_name)
    im_org = median_filt(im_org,3)
    
    min_val = im_org.min()
    max_val = im_org.max()
    min_desired = 12
    max_desired = 230
    print(f"Out float image minimum pixel value: {min_val} and max value: {max_val}")
    out = (max_desired - min_desired) / (max_val - min_val) * (im_org - min_val) + min_desired

    min_val = out.min()
    max_val = out.max()
    print(f"Out float image minimum pixel value: {min_val} and max value: {max_val}")
    print(out[20,20])
    plot_comparison(im_org,out)
# test area, pls ignore
ex2()