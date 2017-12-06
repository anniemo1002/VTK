import numpy as np
import csv
import pylab as pl
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(8, 6))

file = open("unemployment.csv", "rb")
obj = csv.reader(file.read().decode('utf-8').splitlines())

title = False
raw_data = []

for v in obj:
    if title == False:
        title = True
        continue
    if len(v) != 0:
        line_data = [float(v[8]), float(v[9]), float(v[10]),float(v[11])]
        raw_data.append(line_data)

X = PCA(n_components=2).fit_transform(np.array(raw_data))

db = DBSCAN(eps=100, min_samples=7).fit(X)

import matplotlib.pyplot as plt

def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykwrgbrgbrgb')
    target_ids = range(len(target_names))
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(data[target == i, 0], data[target == i, 1],
            c=c, label=label)
    pl.legend()
    pl.show()

plot_2D(X, db.labels_, ["cluster0", "cluster1", "cluster2"])
