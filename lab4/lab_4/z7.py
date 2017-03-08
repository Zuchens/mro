import random
import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


__author__ = 'Zuch'
def annulus(r1,r2):
    r = random.uniform(r1, r2)
    theta = random.uniform(0, 2 * math.pi)
    x = r * math.sin(theta)
    y = r * math.cos(theta)
    return x,y
n =200
X1 = [(annulus(2,4)) for x in range(0,n)]
X2 = [(annulus(5,8)) for x in range(0,n)]
X1.extend(X2)
Y1 = [0 for x in range(0,n)]
Y2 = [1 for x in range(0,n)]
Y1.extend(Y2)

X = np.asarray(X1)
Y = Y1
C = 1.0
h = .02

svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
poly_svc3 = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
poly_svc4 = svm.SVC(kernel='poly', degree=4, C=C).fit(X, Y)
C = 1.0
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial degree 3',
          'SVC with polynomial degree 4']


for i, clf in enumerate((svc,rbf_svc, poly_svc3, poly_svc4)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
