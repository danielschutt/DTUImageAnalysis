import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
# Exercises

# Exercise 2 variance


def var(sep):
    vari = sep.var(ddof=1)
    return vari


# Exercise 3 covariance. High covariance, then low covariance between sepal/petal data.
def cov(a, b, N):
    sum = 0
    for i in range(N):
        sum += a[i]*b[i]
    return (1/(N-1))*sum


# Exercise 4 plotting covaraince plot using seaborn, transform to pandas.
def Transform_to_pandas(data):
    plt.figure()  # Added this to make sure that the figure appear
# Transform the data into a Pandas dataframe
    d = pd.DataFrame(data, columns=['Sepal length', 'Sepal width',
                                    'Petal length', 'Petal width'])
    sns.pairplot(d)
    plt.show()

# Exercise 5, manual PCA analysis, this is covariance matrix. Note manual doesn't work right now, incorrect results. We've transposed a different place since the original dimensions are incorrect for this type of calculation.


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

    d = pd.DataFrame(pc_proj.T, columns=['Sepal length', 'Sepal width',
                                         'Petal length', 'Petal width'])
    sns.pairplot(d)
    plt.show()


# Directory containing data and images
in_dir = "exercises/ex1b-PCA/data/"


txt_name = "irisdata.txt"
iris_data = np.loadtxt(in_dir + txt_name, comments="%")
N = 50

# x is a matrix with 50 rows and 4 columns
x = iris_data[0:50, 0:4]


sep_l = x[:, 0]
sep_w = x[:, 1]
pet_l = x[:, 2]
pet_w = x[:, 3]

# Manual process
Cx = manual_PCA_step1(x, N)
values, vectors = eig(Cx)

examine(values)
pairplots_ex8(x, vectors)

# Automatic process PCA go. The eigenvectors are transposed
print('AUTOMATIC')
pca = decomposition.PCA()
pca.fit(x)
values_pca = pca.explained_variance_
exp_var_ratio = pca.explained_variance_ratio_
vectors_pca = pca.components_

data_transform = pca.transform(x)

print(exp_var_ratio)
print(vectors)
print(vectors_pca.T)
