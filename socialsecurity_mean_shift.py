import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import csv
# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

iris_csv = open("socialsecurity.csv","rb")
iris_ob = csv.reader(iris_csv.read().decode('utf-8').splitlines())
iris_data = []
iris_target = []
b = False
stateCode = {"AL": 0, "AK": 1, "AZ": 2, "AR": 3, "CA": 4, 
             "CO": 5, "CT": 6, "DE": 7, "FL": 8, "GA": 9, 
             "HI": 10, "ID": 11, "IL": 12, "IN": 13, "IA": 14, 
             "KS": 15, "KY": 16, "LA": 17, "ME": 18, "MD": 19, 
             "MA": 20, "MI": 21, "MN": 22, "MS": 23, "MO": 24, 
             "MT": 25, "NE": 26, "NV": 27, "NH": 28, "NJ": 29, 
             "NM": 30, "NY": 31, "NC": 32, "ND": 33, "OH": 34, 
             "OK": 35, "OR": 36, "PA": 37, "RI": 38, "SC": 39, 
             "SD": 40, "TN": 41, "TX": 42, "UT": 43, "VT": 44, 
             "VA": 45, "WA": 46, "WV": 47, "WI": 48, "WY": 49, 
             "DC": 50, "FM": 51, "GU": 52, "MH": 53, "MP": 54, 
             "PW": 55, "PR": 56, "VI": 57, "AE": 58, "AA": 59, 
             "AP": 60 }


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
        line_data = [ float(v[7]), float(v[8]), float(v[10]), float(v[11]), float(v[13]), float(v[15]), float(v[17]), float(v[18]), float(v[20]),float(v[21]), float(v[23]), float(v[25]), float(v[27]), float(v[28])]
    iris_target.append(stateCode[str(v[4]).strip()])
    iris_data.append(line_data)
_ = np.array(iris_target)
data = np.array(iris_data)

X = data[:,:2]
# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=len(data))

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
