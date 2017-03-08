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
from gap_stat import *


class Kmean():
    def __init__(self,K,X):
        self.MAX_I = 100
        self.K = K
        self.X = X

    def find_clusters(self,method):
        new_centers = [random.choice(self.X).tolist() for i in range(0,self.K)]
        old_centers = [[] for i in range(0,self.K)]
        iterations = 0
        while self.has_not_converged(old_centers,new_centers, iterations):
            old_centers = new_centers
            self.create_clusters(new_centers,method)
            new_centers  = self.reeval()
            iterations+=1
        return new_centers,self.clusters



    def create_clusters(self,center_points, method):
        self.clusters = [[] for x in center_points]
        for x in self.X:
            if method == "euclidean":
                min_index, min_value = min(enumerate([scipy.spatial.distance.euclidean(x,center) for center in center_points]), key=operator.itemgetter(1))
            if method == "mahalanobis":
                cov = np.cov(self.X,rowvar=False)
                VI = np.linalg.inv(cov),
                min_index, min_value = min(enumerate([scipy.spatial.distance.mahalanobis(x,center,VI) for center in center_points]), key=operator.itemgetter(1))
            self.clusters[min_index].append(x)

    def reeval(self):
        return [np.mean(i,axis=0)  if not np.isnan(np.mean(i,axis=0)).any() else random.choice(self.X).tolist() for i in self.clusters ]

    def has_not_converged(self, old_centers, new_centers, iterations):
        b = set([tuple(new) for new in new_centers])
        a = set([tuple(old) for old in old_centers])
        return  iterations < self.MAX_I and  a!= b




def load_data():
    mat_contents = sio.loadmat('facesYale.mat')
    X_train =mat_contents['featuresTrain']
    mat = np.copy(mat_contents)
    Y_train = mat_contents['personTrain']
    X_train = np.concatenate((X_train,mat_contents['featuresTest']),axis=0)
    Y_train = np.concatenate((Y_train,mat_contents['personTest']),axis=0)
    return X_train, Y_train

def draw_3d(found_clusters,RGB_tuples,X,Y):
    pca = PCA(n_components=3)

    pca.fit(X)
    print(pca.explained_variance_ratio_)
    clusters = []
    indices = []
    for cluster in found_clusters:
        indices.append([next(idx for idx,i in enumerate(X) if np.array_equal(i,ur)) for ur in cluster])
        clusters.append(pca.transform(cluster))
    HSV_tuples_categories = [(x*1.0/15, random.uniform(0.4,1.0), random.uniform(0.4,1.0)) for x in range(15)]
    RGB_tuples_categories = sorted(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples_categories),key=lambda p:p[0])
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0,len(clusters)):
        ax.scatter([x[0] for x in clusters[i]],[x[1] for x in clusters[i]],[x[2] for x in clusters[i]],c = RGB_tuples_categories[i],s = 150)
        for idx in range(0,len(indices[i])):
            xs = pca.transform(X[indices[i][idx]])[0][0]
            ys = pca.transform(X[indices[i][idx]])[0][1]
            zs = pca.transform(X[indices[i][idx]])[0][2]
            col =Y[indices[i][idx]][0]
            ax.scatter(xs,ys,zs,c = RGB_tuples_categories[col-1], s = 15)

    plt.show()

def draw_2d(found_clusters,RGB_tuples,X,Y):
    pca = PCA(n_components=3)
    pca.fit(X)
    clusters = []
    indices = []
    import colorsys
    HSV_tuples_categories = [(x*1.0/15, random.uniform(0.4,1.0), random.uniform(0.4,1.0)) for x in range(15)]
    RGB_tuples_categories = sorted(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples_categories),key=lambda p:p[0])
    for cluster in found_clusters:
        indices.append([next(idx for idx,i in enumerate(X) if np.array_equal(i,ur)) for ur in cluster])
        clusters.append(pca.transform(cluster))
    for i in range(0,len(clusters)):
        plt.scatter([x[0] for x in clusters[i]],[x[1] for x in clusters[i]],c = RGB_tuples[i],s = 150)
        for idx in range(0,len(indices[i])):
            xs = pca.transform(X[indices[i][idx]])[0][0]
            ys = pca.transform(X[indices[i][idx]])[0][1]
            col =Y[indices[i][idx]][0]
            plt.scatter(xs,ys,c = RGB_tuples_categories[col-1], s = 15)
    plt.show()



if __name__ == "__main__":
    X,Y = load_data()
    K = 5
    kmean = Kmean(K=K,X=X)
    mu,found_clusters = kmean.find_clusters("euclidean")
    import colorsys
    HSV_tuples = [(x*1.0/K, random.uniform(0.4,1.0), random.uniform(0.4,1.0)) for x in range(K)]
    RGB_tuples = sorted(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples),key=lambda p:p[0])

    draw_2d(found_clusters,RGB_tuples,X,Y)
    draw_3d(found_clusters,RGB_tuples,X,Y)
    ks, logWks, logWkbs, sk,gaps = gap_statistic(X)
    gaps = gaps.tolist()
    sk = sk.tolist()
    res = [0]*15
    for i in range(0,len(gaps)-2):
        res[i] = gaps[i]-gaps[i+1]+ sk[i+1]
    plt.bar(ks,gaps)
    plt.show()
    X,Y = load_data()