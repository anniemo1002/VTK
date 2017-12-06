import csv
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

file = open("socialsecurity.csv", "rb")
obj = csv.reader(file.read().decode('utf-8').splitlines())

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

title = False
raw_data = []
raw_target = []

for v in obj:
	if title == False:
		title = True
		continue
	if len(v) != 0:
	    line_data = [float(v[6]), float(v[7]), float(v[8]), float(v[10]), float(v[11]), float(v[13]), float(v[15]), float(v[17]), float(v[18]), float(v[20]), float(v[21]), float(v[23]), float(v[25]), float(v[27]), float(v[28])]
	    raw_data.append(line_data)
	    raw_target.append(stateCode[str(v[4]).strip()])

data = numpy.array(raw_data)
target = numpy.array(raw_target)

X = data[:, :3]
y = target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(1, figsize=(8, 6))
plt.clf()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=4).fit_transform(data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],X_reduced[:,3], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
