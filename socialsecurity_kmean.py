import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
#X, y = make_blobs(n_samples=n_samples, random_state=random_state)

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
    arr = [float(v[0]),float(v[1]),float(v[2]),float(v[3])]'''
    if len(v) != 0:
        line_data = [ float(v[7]), float(v[8]), float(v[10]), float(v[11]), float(v[13]), float(v[15]), float(v[17]), float(v[18]), float(v[20]),float(v[21]), float(v[23]), float(v[25]), float(v[27]), float(v[28])]
    iris_target.append(stateCode[str(v[4]).strip()])
    iris_data.append(line_data)
y = np.array(iris_target)
data = np.array(iris_data)
n_sample = len(data)
X = data[:,:2]
# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Incorrect Number of Blobs")

# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

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
