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

from skimage.color import rgb2hsv
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

#Exercises from here!
def accImage():
    #Answer is 270. Path is 30, 192, 48,9,112. Added 
    print("HI")

def aorta():
    in_dir = "exercises/spring 2022/data/Aorta/"
    ct = dicom.read_file(in_dir + '1-442.dcm')
    img = ct.pixel_array

    aorta_roi = io.imread(in_dir + "AortaROI.png")
    aorta_mask = aorta_roi > 0
    aorta_values = img[aorta_mask]
    (mu_aorta, std_aorta) = norm.fit(aorta_values)

    back_roi = io.imread(in_dir + "BackROI.png")
    back_mask = back_roi > 0
    back_values = img[back_mask]
    (mu_back, std_back) = norm.fit(back_values)

    liver_roi = io.imread(in_dir + "LiverROI.png")
    liver_mask = liver_roi > 0
    liver_values = img[liver_mask]
    (mu_liver, std_liver) = norm.fit(liver_values)
    
    min_hu = 140
    max_hu = 160
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_back = norm.pdf(hu_range, mu_back, std_back)
    pdf_aorta = norm.pdf(hu_range, mu_aorta, std_aorta)
    pdf_liver = norm.pdf(hu_range, mu_liver, std_liver)

    #Parametric classifier. Read plot
    
    plt.plot(hu_range, pdf_aorta, 'g--', label="aorta")
    plt.plot(hu_range, pdf_liver, label="liver")
    plt.title("Fitted Gaussians")
    plt.legend()
    #plt.show()

    #Second part from here
    
    img[img > 90] = 255
    img[img < 90] = 0
    
    #Convert dicom to grayscale
    img_2d = img.astype(float)
    img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
    img = np.uint8(img_2d_scaled)
    
    img = clear_border(img)
    img = img_as_float(img)
    label_img = measure.label(img, connectivity=2)
    #plot_comparison(label_img,img)
    region_props = measure.regionprops(label_img)
    
    
    areas = np.array([prop.area for prop in region_props])
    peris = np.array([prop.perimeter for prop in region_props])
    circularity = (4*math.pi*areas)/(np.power(peris, 2))
    
    #Don't know why i had to change circularity. The region had a circularity of 0.948 or simething.
    min_circ = 0.90
    circ_mask = circularity > min_circ
    

    min_area = 200
    area_mask = areas > min_area
    
    counter = 0
    circCounter = 0
    for i in range(0, len(circ_mask)):
        if area_mask[i]:
            if circ_mask[i]:
                counter = counter+1
                
                print(len(region_props[i].coords)*(0.75*0.75))
    print(mu_aorta)
    print(std_aorta) 

def cars():
    in_dir = "exercises/spring 2022/data/CarData/"
    cars = io.imread(in_dir + "car.png")
    cars = rgb2hsv(cars)
    h_comp = cars[:, :, 0]
    s_comp = cars[:, :, 1]
    v_comp = cars[:, :, 2]
    
    segm = (s_comp >0.7)   
    
    cars[segm] = 255
    cars[~segm] = 0

    cars = gray(cars)
    
    print(cars)
    dis = disk(6)
    
    cars = erosion(cars,dis)

    dis = disk(4)
    cars = dilation(cars,dis)
    
    #EXAM ANSWER
    #print(np.sum(cars > 0))
    #NExt part road
    in_dir = "exercises/spring 2022/data/CarData/"
    road = io.imread(in_dir + "road.png")
    road = rgb2hsv(road)
    h_comp = road[:, :, 0]
    s_comp = road[:, :, 1]
    v_comp = road[:, :, 2]
    
    segm = (v_comp >0.9)   
    
    road[segm] = 1.0
    road[~segm] = 0.0
    #Pas på med at bruge gray
    #road = gray(road)

    label_img = measure.label(road, connectivity=2)
    label_img_filter = label_img
    #plot_comparison(label_img,img)
    region_props = measure.regionprops(label_img)
    #EXAM ANSWER
    min_area = 1047
    
    counter = 0
    for region in region_props:
        if region.area > min_area:
            # set the pixels in the invalid areas to background
            
            counter = counter +1
        else:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    plot_comparison(road,label_img_filter)
    print(counter)

def SPOONS():
    in_dir = "exercises/spring 2022/data/ImagePCA/"
    spoon1 = "spoon1.png"
    spoon2 = "spoon2.png"
    spoon3 = "spoon3.png"
    spoon4 = "spoon4.png"
    spoon5 = "spoon5.png"
    spoon6 = "spoon6.png"

    spoon1 = io.imread(in_dir + spoon1)
    spoon2 = io.imread(in_dir + spoon2)
    spoon3 = io.imread(in_dir + spoon3)
    spoon4 = io.imread(in_dir + spoon4)
    spoon5 = io.imread(in_dir + spoon5)
    spoon6 = io.imread(in_dir + spoon6)


    blend = 1/6 * img_as_float(spoon1) + 1/6 * img_as_float(spoon2) + 1/6 * img_as_float(spoon3) + 1/6 * img_as_float(spoon4) + 1/6 * img_as_float(spoon5) + 1/6 * img_as_float(spoon6)
    
    im = thresh(blend,100)

    mn = np.mean(im)
    data = im - mn
    data = data / data.std(axis=0)
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

    blend = img_as_ubyte(blend)
    print(blend[500,100])


def POINTS():
    deg = 20
    a = simTrans((np.array([[10],[10]])),20,[3.1,-3.3],2)
    print(a)

def SOCCER():
    in_dir = "exercises/spring 2022/data/PCAData/"
    txt_name = "soccer_data.txt"
    car_data = np.loadtxt(in_dir + txt_name, comments="%")

    x = car_data
    n_feat = x.shape[1]
    n_obs = x.shape[0]
    print(f"Number of features: {n_feat} and number of observations: {n_obs}")


    mn = np.mean(x, axis=0)
    data = x - mn
    data = data / data.std(axis=0)
    c_x = np.cov(data.T)


    values, vectors = np.linalg.eig(c_x)
    


    pc_proj = vectors.T.dot(x.T)
    #Hella wrong answer for PCA answer 2
    print(pc_proj.flat[np.abs(pc_proj).argmax()])
    print(pc_proj.flat[np.abs(pc_proj).argmax()])

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


def GRAIN():
    mn_bad = 25
    std_bad = 10*10

    mn_med = 52
    std_med = 2*2
    
    mn_god = 150
    std_god = 30*30
    hu_range = np.arange(0,200,1.0)
    pdf_bad = norm.pdf(hu_range, mn_bad, std_bad)
    pdf_med = norm.pdf( hu_range,mn_med, std_med)
    pdf_god = norm.pdf( hu_range,mn_god, std_god)

    #Parametric classifier. Read plot
    
    plt.plot( hu_range,pdf_med, 'r--', label="med")
    plt.plot( hu_range,pdf_god, label="good")
    plt.title("Fitted Gaussians")
    plt.legend()
    plt.show()

    print((mn_bad + mn_med)/2)
def volcano():
    passive = np.array([[1.2,1.1],[2.9,0.4],[1.7,-2.7],[1.8,-0.3],[3.2,1.3],[3.1,-0.9],[0.5,1.7],[1.4,-2.1],[2.7,-0.8],[2.0,0.5]])
    active = np.array([[0.5,1.7],[1.4,-2.1],[2.7,-0.8],[2.0,0.5]])

    X = np.r_[active]
    #W = LDA(X, T)
    X1=np.array([1.2,2.9,1.7,1.8,3.2,3.1])
    Y1=np.array([1.1,0.4,-2.7,-0.3,1.3,-0.9])
    X2=np.array([0.5,1.4,2.7,2])
    Y2=np.array([1.7,-2.1,-0.8,0.5])

    
    Input=[[X1,X2],[Y1,Y2]]
    Input = passive
    Target=np.array([[0],[0],[0],[0],[0],[0],[1],[1],[1],[1]])
    

    W = LDA(passive,Target)


def landmarks():
    in_dir = "exercises/spring 2022/data/Landmarks/"
    play1 = "play1.png"
    play1 = io.imread(in_dir + play1)

    play5 = "play5.png"
    play5 = io.imread(in_dir + play5)

    ref = "reference.png"
    ref = io.imread(in_dir + ref)

    fixed = loadmat('exercises/spring 2022/data/Landmarks/playfixedPoints.mat')['fixedPoints']
    moving = loadmat('exercises/spring 2022/data/Landmarks/playmovingPoints.mat')['movingPoints']
    
    
    moved = moveEst(play5,moving,fixed)
    
    threshd = np.invert(threshold_image(moved,180,True))
    plot_comparison(moved,threshd)
    print(len(threshd))
    print(len(ref))
    #print(DICE_COE(threshd,ref))
    print(math.dist(fixed[0],moving[0]))

#Test area, pls ignore

cars()