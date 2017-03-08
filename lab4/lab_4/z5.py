import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def make_points_L(n,max_x):
    X = [random.uniform(0,max_x) for i in range(0,n)]
    return [[x+2,random.uniform(0,x)+1] for x in X]

def make_points_U(n,max_x):
    X = [random.uniform(0,max_x) for i in range(0,n)]
    return [[max_x - x,max_x - random.uniform(0,x)] for x in X]

def make_points():
    points = make_points_L(30, 5)
    plt.scatter([x[0] for x in points],[x[1] for x in points], c='r')
    points2 = make_points_U(30, 5)
    plt.scatter([x[0] for x in points2],[x[1] for x in points2])
    Y1 = [0]*len(points)
    Y2 = [1]*len(points2)
    Y1.extend(Y2)
    points.extend(points2)
    y = np.asarray(Y1)
    X = np.asarray(points)
    return X,y

def fit_svc(X,y):
    return svm.SVC(kernel='linear').fit(X, y)

def plot_sep(x_min,x_max,svc):
    w = svc.coef_[0]
    a = -w[0] / w[1]
    xx2 = np.linspace(x_min, x_max)
    yy2 = a * xx2 - (svc.intercept_[0]) / w[1]
    b = svc.support_vectors_[0]
    yy_down = a * xx2 + (b[1] - a * b[0])
    b = svc.support_vectors_[-1]
    yy_up = a * xx2 + (b[1] - a * b[0])
    plt.plot(xx2, yy2, 'k-')
    plt.plot(xx2, yy_down, 'k--')
    plt.plot(xx2, yy_up, 'k--')


def plot_linear_svc(svc,X,y):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plot_sep(x_min,x_max,svc)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], c='g', cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())


X,y = make_points()
svc = fit_svc(X,y)
plot_linear_svc(svc,X,y)
plt.show()