import time

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA

# #############################################################################
# Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
#X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

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
labels_true = np.array(iris_target)
data = np.array(iris_data)
n_sample = len(data)
X = PCA(n_components=2).fit_transform(data)
#X = data[:,:2]
# Incorrect number of clusters
# #############################################################################
# Compute clustering with Means

k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
t0 = time.time()
k_means.fit(X)
t_batch = time.time() - t0

# #############################################################################
# Compute clustering with MiniBatchKMeans

mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0

# #############################################################################
# Plot result

fig = plt.figure(figsize=(8, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers,
                                          mbk_means_cluster_centers)


# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %(t_mini_batch, mbk.inertia_))
plt.show()
