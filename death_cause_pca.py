
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import urllib.request
import numpy
import csv
# Download the data
#url_str = 'http://blender.chrisconlan.com/iris.csv'
iris_csv = open("death_cause.csv","rb")
  
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
    if v[2] =="Unintentional Injuries":
        iris_target.append(0)
    elif v[2] =="All Causes":
        iris_target.append(1)
    elif v[2] =="Alzheimer's disease":
        iris_target.append(2)
    elif v[2]=="Homicide":
        iris_target.append(3)
    elif v[2]=="Stroke":
        iris_target.append(4)
    elif v[2]=="Chronic liver disease and cirrhosis":
        iris_target.append(5)
    elif v[2] =="CLRD":
        iris_target.append(6)
    elif v[2] =="Diabetes":
        iris_target.append(7)
    elif v[2]=="Diseases of Heart":
        iris_target.append(8)
    elif v[2]=="Essential hypertension and hypertensive renal disease":
        iris_target.append(9)
    elif v[2]=="Influenza and pneumonia":
        iris_target.append(10)
    elif v[2]=="Cancer":
        iris_target.append(11)
    elif v[2] =="Suicide":
        iris_target.append(12)
    elif v[2]=="Kidney Disease":
        iris_target.append(13)
    elif v[2] =="Parkinson's disease":
        iris_target.append(14)
    elif v[2] =="Pneumonitis due to solids and liquids":
        iris_target.append(15)
    elif v[2] =="Septicemia":
        iris_target.append(16)
    arr = [float(v[0]),float(v[4]),float(v[5])]
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
plt.xlabel('year')
plt.ylabel('type')

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
