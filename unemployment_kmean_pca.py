import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
#X, y = make_blobs(n_samples=n_samples, random_state=random_state)

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
y = np.array(target)
data = np.array(data)
n_sample = len(data)
X= PCA(n_components=2).fit_transform(data)
# Incorrect number of clusters
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=4, random_state=random_state).fit_predict(X_aniso)

plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# Different variance
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                        cluster_std=[1.0, 2.5, 0.5],
                                                                        random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# Unevenly sized blobs
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                        random_state=random_state).fit_predict(X_filtered)

plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
