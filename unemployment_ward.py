import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import numpy
import csv
from sklearn.decomposition import PCA
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

#digits = datasets.load_digits(n_class=10)
#X = digits.data

# Number of samples per component
#n_samples = 500

iris_csv = open("unemployment.csv","rb")
iris_ob = csv.reader(iris_csv.read().decode('utf-8').splitlines())
data = []
target = []
b = False
for v in iris_ob:
    if b == False:
        b = True
        continue
    '''if v[4] =="setosa":
        iris_target.append(0)
    elif v[4] == "versicolor":
        iris_target.append(1)
    elif v[4] == "virginica":
        iris_target.append(2)
    arr = [float(v[0]),float(v[1]),float(v[2]),float(v[3])]
    iris_data.append(arr)'''
    if len(v) != 0:
        line_data = [float(v[8]), float(v[9]), float(v[10])]
        data.append(line_data)
        target.append(int(v[2]))
y= numpy.array(target)
data = numpy.array(data)

X = data

#X= PCA(n_components=2).fit_transform(data)

# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

#y = digits.target
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
        # Having a larger dataset shows more clearly the behavior of the
            # methods, but we multiply the size of the dataset only by 2, as the
                # cost of the hierarchical clustering methods are strongly
                    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),.3 * np.random.normal(size=2),mode='constant',).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, X, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),color=plt.cm.spectral(labels[i] / 10.),fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s : %.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, X, clustering.labels_, "%s linkage" % linkage)


plt.show()
