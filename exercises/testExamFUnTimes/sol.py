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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize


def manual_PCA_step1(x, N):
    mn = np.mean(x, axis=0)
    data = x - mn
    Cx = (1/(N-1))*np.matmul(data.T, data)

    # print(Cx)
    # print("PROPER")
    # print(np.cov(x.T))
    return Cx


# Exercise 6, eigenvectors and eigenvalues. Still manual PCA
def eig(Cx):
    values, vectors = np.linalg.eig(Cx)  # Here c_x is your covariance matrix.
    return (values, vectors)


# exercise 7, still manual PCA. examine Principal components
def examine(values):
    v_norm = values / values.sum() * 100
    plt.plot(v_norm)
    plt.xlabel('Principal component')
    plt.ylabel('Percent explained variance')
    plt.ylim([0, 100])
    plt.xlim([0, 4])

    plt.show()


# Exericse 8, seaborn pair-plots. This is no longer anything covariant?
def pairplots_ex8(x, vectors):
    mn = np.mean(x, axis=0)
    data = x - mn
    pc_proj = vectors.T.dot(data.T)
    # print(pc_proj.T)
    # Transform the data into a Pandas dataframe

    d = pd.DataFrame(pc_proj.T[:, 0:3], columns=[' 1', '2',
                                                 '3'])
    sns.pairplot(d)
    plt.show()


def gray(input_img):
    new_img = color.rgb2gray(input_img)
    return new_img


def stretch(img, min, max):
    img = color.rgb2gray(img)
    min_val = img.min()
    max_val = img.max()

    print(
        f"Float image minimum pixel value: {min_val} and max value: {max_val}")

    min_desired = 0.1
    max_desired = 0.6

    img_out = (max_desired - min_desired) / (max_val - min_val) * \
        (img - min_val) + min_desired

    return img_out


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
    print("otsu's threshold " + str(otsu_thresh))
    return img_as_ubyte(thresh(im_float, otsu_thresh))


def calcAlignment(src, dst):
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    print(f"Landmark alignment error F: {f}")
    return f


def moveEst(src_img, src, dst):
    tform = EuclideanTransform()
    tform.estimate(src, dst)
    src_transform = matrix_transform(src, tform.params)
    return warp(src_img, tform.inverse)


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def show(input_img):
    io.imshow(input_img)
    io.show()


def gaussian_filt(input_img, sigma):
    gauss_img = gaussian(input_img, sigma)
    return gauss_img


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


def gss_multiple(data1, data2, label1="Data1", label2="data2", tit="Comparison of gaussians"):
    min_hu = -200
    max_hu = 1000
    hu_range = np.arange(min_hu, max_hu, 1.0)
    pdf_spleen = norm.pdf(hu_range, np.mean(data1), np.std(data1))
    pdf_bone = norm.pdf(hu_range, np.mean(data2), np.std(data2))
    print(pdf_spleen[38])
    print(pdf_bone[38])
    plt.plot(hu_range, pdf_spleen, 'r-', label=label1)
    plt.plot(hu_range, pdf_bone, 'b', label=label2)
    plt.title(tit)
    plt.legend()
    plt.show()
# Solutions here


def ex1():
    im_name = "rocket.png"
    im_org = io.imread(in_dir + im_name)
    im_gauss = gaussian_filt(im_org, 3)

    fin_pic = img_as_ubyte(im_gauss)
    print(fin_pic[100][100])


def ex2():
    im_name = "rocket.png"
    im_org = io.imread(in_dir + im_name)
    prew = prewitt_filt(im_org)
    thresh = threshold_image(prew, 0.06)
    show(thresh)
    print(np.sum(thresh == 1.0))


def ex4():
    cow = [26, 46, 33, 23, 35, 28, 21, 30, 38, 43]
    sheep = [67, 27, 40, 60, 39, 45, 27, 67, 43, 50, 37, 100]
    avg_cow = np.mean(cow)
    avg_sheep = np.mean(sheep)
    print("Avg cow" + str(avg_cow) + " avg sheep " + str(avg_sheep))
    print("thresh ")
    print((avg_cow+avg_sheep)/2.0)
    gss_multiple(cow, sheep)

    print(normpdf(38, np.mean(cow), np.std(cow)))

    print(normpdf(38, np.mean(sheep), np.std(sheep)))


def ex5():
    in_dir = "exercises/testExamFUnTimes/data/"
    im_name = "rocket.png"
    im_org = io.imread(in_dir + im_name)
    src = np.array([[220, 55], [105, 675], [315, 675]])
    dst = np.array([[100, 165], [200, 605], [379, 525]])
    print("Alignment before" + str(calcAlignment(src, dst)))
    moved = moveEst(im_org, src, dst)
    moved = img_as_ubyte(moved)
    print(moved[150, 150])

    # What is the pixel value at (row=150, column=150) in the warped image? 129
    # What is the landmark alignment error, F, before the registration? 67021
    # What is the landmark alignment error, F, after the registration?


def ex7():
    in_dir = "exercises/testExamFUnTimes/data/"
    im_name = "figures.png"
    im_org = io.imread(in_dir + im_name)
    im = (gray(im_org))
    ots = np.invert(otsu(im))
    label_img = measure.label(ots)
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    peri = np.array([prop.perimeter for prop in region_props])
    print(areas)
    print(areas[24])
    print(peri[24])
    print(peri)
    show(ots)


def ex8():
    in_dir = "exercises/testExamFUnTimes/data/"
    im_name = "car_data.txt"
    car_data_org = np.loadtxt(in_dir + im_name, comments="%")
    car_data = normalize(car_data_org, axis=0, norm='l1')
    pca = decomposition.PCA()
    pca.fit(car_data)
    values_pca = pca.explained_variance_
    exp_var_ratio = pca.explained_variance_ratio_
    vectors_pca = pca.components_
    data_transform = pca.transform(car_data)
    print(exp_var_ratio[0]+exp_var_ratio[1])  # answer to question 8

    # answer 9

    X = car_data_org
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    print(X[0][0])


def ex13():
    in_dir = "exercises/testExamFUnTimes/data/"
    im_name = "pixelwise.png"
    im_org = io.imread(in_dir + im_name)
    im = stretch(im_org, 0.1, 0.6)
    im = (otsu(im))
    show(im)


def ex11():
    in_dir = "exercises/testExamFUnTimes/data/"
    im_name = "car_data.txt"
    car_data_org = np.loadtxt(in_dir + im_name, comments="%")
    N = car_data_org.shape[0]

    Cx = manual_PCA_step1(car_data_org, N)
    values, vectors = eig(Cx)
    pairplots_ex8(car_data_org, vectors)


# test area, pls ignore
in_dir = "exercises/testExamFUnTimes/data/"

# X-ray image
# im_name = "DTUSign1.jpg"
im_name = "rocket.png"
im_org = io.imread(in_dir + im_name)

ex2()
