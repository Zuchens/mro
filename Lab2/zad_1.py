# -*- coding: utf-8 -*-
import math
from matplotlib import cm
from matplotlib.collections import PolyCollection
import numpy as np
import scipy
import scipy.io as sio
import random
import operator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Kmean():
    def __init__(self,K,X):
        self.MAX_I = 100
        self.K = K
        self.X = X

    def find_clusters(self):
        new_centers = [random.choice(self.X).tolist() for i in range(0,self.K)]
        old_centers = [[] for i in range(0,self.K)]
        iterations = 0
        while self.has_not_converged(old_centers,new_centers, iterations):
            old_centers = new_centers
            self.create_clusters(new_centers)
            new_centers  = self.reeval()
            iterations+=1
        return self.clusters



    def create_clusters(self,center_points):
        self.clusters = [[] for x in center_points]
        for x in self.X:
            min_index, min_value = min(enumerate([scipy.spatial.distance.euclidean(x,center) for center in center_points]), key=operator.itemgetter(1))
            self.clusters[min_index].append(x)

    def reeval(self):
        return [np.mean(i,axis=0)  if not np.isnan(np.mean(i,axis=0)).any() else random.choice(self.X).tolist() for i in self.clusters ]

    def has_not_converged(self, old_centers, new_centers, iterations):
        b = []
        a = set([tuple(old) for old in old_centers])
        try:
            b =set([tuple(new) for new in new_centers])
        except:
            c = [x for x in new_centers if  isinstance(x, float)]
            d=1
        return  iterations < self.MAX_I and  a!= b



def load_data():
    mat_contents = sio.loadmat('facesYale.mat')
    X_train =mat_contents['featuresTrain']
    mat = np.copy(mat_contents)
    # Y_train = [x[0] for x in mat_contents['personTrain']]
    Y_train = mat_contents['personTrain']
    X_train = np.concatenate((X_train,mat_contents['featuresTest']),axis=0)
    Y_train = np.concatenate((Y_train,mat_contents['personTest']),axis=0)
    return X_train, Y_train

def create_polygon(data_x,data_y):

    # Generate data. In this case, we'll make a bunch of center-points and generate
    # verticies by subtracting random offsets from those center-points
    numpoly, numverts = 9, 15
    changed = np.where(data_y[:-1] != data_y[1:])[0]
    changed  = [i+1 for i in changed]
    poly = np.split(data_x, changed)
    centers = 100 * (np.random.random((numpoly,2)) - 0.5)
    offsets = 10 * (np.random.random((numverts,numpoly,2)) - 0.5)
    verts = centers + offsets
    verts = np.swapaxes(verts, 0, 1)

    # In your case, "verts" might be something like:
    # verts = zip(zip(lon1, lat1), zip(lon2, lat2), ...)
    # If "data" in your case is a numpy array, there are cleaner ways to reorder
    # things to suit.

    # Color scalar...
    # If you have rgb values in your "colorval" array, you could just pass them
    # in as "facecolors=colorval" when you create the PolyCollection
    z = np.random.random(numpoly) * 500

    fig, ax = plt.subplots()

    # Make the collection and add it to the plot.
    coll = PolyCollection(poly, array=z, cmap=cm.jet, edgecolors='none')
    ax.add_collection(coll)
    ax.autoscale_view()

    # Add a colorbar for the PolyCollection
    fig.colorbar(coll, ax=ax)
    # plt.show()


X,Y = load_data()
# for x in X:
pca = PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
# create_polygon( np.copy(X), np.copy(Y))
kmean = Kmean(K=15,X=X)
clusters = kmean.find_clusters()
colors = ['#ff0000','#00ff00','#0000ff','#ffff00','#ff00ff','#00ffff','#ccff00','#cc00cc','#ffffff','#ccff00']

import colorsys
N =15
HSV_tuples = [(x*1.0/N, random.uniform(0.4,1.0), random.uniform(0.4,1.0)) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

#
# for i in range(0,len(clusters)):
#     plt.scatter([x[0] for x in clusters[i]],[x[1] for x in clusters[i]],c = RGB_tuples[i],s = 150)
#
# for x_pca,y in zip(X,Y):
#     i = y[0]-1
#     plt.scatter(x_pca[0],x_pca[1],c = RGB_tuples[i], s = 15)
# plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0,len(clusters)):
    ax.scatter([x[0] for x in clusters[i]],[x[1] for x in clusters[i]], [x[2] for x in clusters[i]],c = RGB_tuples[i],s = 150)

for x_pca,y in zip(X,Y):
    i = y[0]-1
    ax.scatter(x_pca[0],x_pca[1],x_pca[2],c = RGB_tuples[i], s = 15)

# plt.ylim([0,100])
plt.show()