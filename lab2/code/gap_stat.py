import random
from numpy.ma import zeros
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import matplotlib.pyplot as plt
import numpy as np

from zad_1 import Kmean
dst = scipy.spatial.distance.euclidean


import scipy.io as sio


def load_data():
    mat_contents = sio.loadmat('facesYale.mat')
    X_train =mat_contents['featuresTrain']
    mat = np.copy(mat_contents)
    Y_train = mat_contents['personTrain']
    X_train = np.concatenate((X_train,mat_contents['featuresTest']),axis=0)
    Y_train = np.concatenate((Y_train,mat_contents['personTest']),axis=0)
    return X_train, Y_train

def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])

def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)

def gap_statistic(X, ):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    # Dispersion for real distribution
    ks = range(1,15)
    Wks = np.zeros(len(ks))
    gaps = np.zeros(len(ks))
    Wkbs = np.zeros(len(ks))
    sk = np.zeros(len(ks))
    for indk, k in enumerate(ks):
        kmean = Kmean(K=k,X=X)
        mu,clusters = kmean.find_clusters("euclidean")
        Wks[indk] = np.log(Wk(mu, clusters))
        # Create B reference datasets
        B = 15
        BWkbs = zeros(B)
        for i in range(B):
            Xb = []
            for n in range(len(X)):
                Xb.append([random.uniform(xmin,xmax),
                          random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            kmean = Kmean(K=k,X=Xb)
            mu,clusters = kmean.find_clusters("euclidean")
            BWkbs[i] = np.log(Wk(mu, clusters))
        Wkbs[indk] = sum(BWkbs)/B
        sk[indk] = np.sqrt(sum((BWkbs-Wkbs[indk])**2)/B)
        gaps[indk] = sum(BWkbs-Wkbs[indk])/B
    sk = sk*np.sqrt(1+1/B)
    return(ks, Wks, Wkbs, sk,gaps)


