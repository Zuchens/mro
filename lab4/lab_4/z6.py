import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors


def neighbours_test(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print 'Neighbors'
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    return neigh

def svc_test(X_train, X_test, y_train, y_test):
    svc = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print 'SVC with linear kernel'
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    return svc

def rbf_svc_test(X_train, X_test, y_train, y_test):
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7).fit(X_train, y_train)
    y_pred = rbf_svc.predict(X_test)
    print 'SVC with RBF kernel'
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    return rbf_svc

def poly_svc_test(X_train, X_test, y_train, y_test):
    poly_svc = svm.SVC(kernel='poly', degree=3).fit(X_train, y_train)
    print 'SVC with polynomial (degree 3) kernel'
    y_pred = poly_svc.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    return poly_svc

def plot_results(svc,neigh,rbf_svc, poly_svc,X,y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    titles = ['SVC with linear kernel',
              'Neighbors',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']


    for i, clf in enumerate((svc,neigh,rbf_svc, poly_svc)):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

def get_class(X_train, X_test, y_train, y_test):
    neigh = neighbours_test(X_train, X_test, y_train, y_test)
    svc = svc_test(X_train, X_test, y_train, y_test)
    rbf_svc = rbf_svc_test(X_train, X_test, y_train, y_test)
    poly_svc = poly_svc_test(X_train, X_test, y_train, y_test)
    return svc,neigh,rbf_svc, poly_svc
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
svc,neigh,rbf_svc, poly_svc = get_class(X_train, X_test, y_train, y_test)
pca = PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
svc,neigh,rbf_svc, poly_svc = get_class(X_train, X_test, y_train, y_test)
h = .02

plot_results(svc,neigh,rbf_svc, poly_svc,X,y)
