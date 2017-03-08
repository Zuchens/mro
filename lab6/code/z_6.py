from PIL import Image
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid

def load_cifar():
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    import numpy as np
    X_train  = []
    y_train  = []
    for i in range(1,6):
        pickled = unpickle('cifar10\\data_batch_'+ str(i))
        if X_train == []:
            X_train = pickled['data']
        else:
            X_train = np.append(X_train,pickled['data'],axis=0)
        y_train.extend(pickled['labels'])

    pickled = unpickle('cifar10\\test_batch')
    X_test = pickled['data']
    y_test = pickled['labels']
    # X_train, y_train = X_train[:20000],y_train[:20000]
    # X_test_rows, y_test = X_test_rows[:500],y_test[:500]
    X_train = preprocessing.scale(X_train)
    X_test =preprocessing.scale(X_test)
    return 32,X_train,X_test,y_train,y_test

def load_mnist():
    mnist = fetch_mldata('MNIST original')
    X = mnist['data']
    y = mnist['target']
    X_train,X_test = X[:60000],X[60000:]
    y_train,y_test = y[:60000],y[60000:]
    # X_train = preprocessing.scale(X_train)
    # X_test=preprocessing.scale(X_test)
    return 28,X_train,X_test,y_train,y_test

print "Load dataset"
size,X_train,X_test,y_train,y_test = load_mnist()
print len(X_train),len(X_test)
print "SGD: warstwy 50 basic"
clf = MLPClassifier(solver='sgd', alpha=0.0001,
                    hidden_layer_sizes=(50, ),max_iter=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)


print "SGD: warstwy 100"
clf = MLPClassifier(solver='sgd', alpha=0.0001,
                    hidden_layer_sizes=(100,),max_iter=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
# plt.plot(clf.loss_curve_,'r-')

# fig, axes = plt.subplots(10, 10)
# vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
# for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
#     # r = coef[:1024]
#     # g = coef[1024:2048]
#     # b = coef[-1024:]
#     #
#     # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     # ax.matshow(gray.reshape(32,32), cmap=plt.cm.gray, vmin=.5 * vmin,
#     ax.matshow(coef.reshape(28,28), cmap=plt.cm.gray, vmin=.5 * vmin,
#     vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
# plt.show()
print "SGD: warstwy 30,10"
clf = MLPClassifier(solver='sgd', alpha=0.0001,
                    hidden_layer_sizes=(30,10 ),max_iter=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
plt.plot(clf.loss_curve_,'b-')

print "SGD: warstwy 50,30,10"
clf = MLPClassifier(solver='sgd', alpha=0.0001,
                    hidden_layer_sizes=(50,30,10 ),max_iter=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
plt.plot(clf.loss_curve_,'k-')
plt.show()

print "LBFGS: warstwy 50,30,10"
clf = MLPClassifier(solver='lbfgs', alpha=0.0001,
                    hidden_layer_sizes=(50,30,10 ),max_iter=30)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)


print "1NN"
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print accuracy_score(y_test, y_pred)

print "Closest centroid"
clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print accuracy_score(y_test, y_pred)
# print classification_report(y_test, y_pred)

print "SGD loss curve"
clf = MLPClassifier(solver='sgd', alpha=0.0001,hidden_layer_sizes=(50, ),max_iter=30)
clf.fit(X_train, y_train)
plt.plot(clf.loss_curve_)
plt.show()

fig, ax = plt.subplots()
fig.canvas.draw()
accuracies_alpha = []
min_alpha = 0.000000001
for i in range(1,10):
    clf = MLPClassifier(solver='sgd', alpha= min_alpha*pow(10,i),
                    hidden_layer_sizes=(50, ),max_iter=30)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies_alpha.append(accuracy_score(y_test, y_pred))
ax.plot([i for i in range(0,len(accuracies_alpha))],accuracies_alpha)
ax.set_xticklabels([str(min_alpha*pow(10,i+1)) for i in range(0,len(accuracies_alpha))])
plt.show()