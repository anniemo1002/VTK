
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import urllib.request
import numpy
import csv
# Download the data
url_str = 'http://blender.chrisconlan.com/iris.csv'
iris_csv = open("diabetes.csv","rb")
  
# Read it into a csv iterator
iris_ob = csv.reader(iris_csv.read().decode('utf-8').splitlines())
# import some data to play with
#iris = datasets.load_iris()
#X = iris.data[:, :2]  # we only take the first two features.
#y = iris.target
iris_data = []
iris_target = []
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
        iris_target.append(2)'''
    iris_target.append(float(v[4]))
    arr = [float(v[0]),float(v[1]),float(v[2]),float(v[3])]
    iris_data.append(arr)
y = numpy.array(iris_target)
data = numpy.array(iris_data)

X = data[:,:2]
x_min= X[:, 0].min() - .5
x_max = X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('CT')
plt.ylabel('BPAD')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris_data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()
