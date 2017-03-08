from __future__ import print_function
import copy
import os, os.path
from PIL import Image
import numpy as np
from sklearn.cross_validation import train_test_split


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_image( infilename ) :
    image = Image.open(infilename)
    img_grey = image.convert('L')
    img = np.array(img_grey, dtype=np.float)
    return img

def get_70_faces_dir(dir):
    X = []
    Y = []
    i = 0
    names = []
    for name in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, name)):
            files =[name_file for name_file in os.listdir(dir+"/"+name) if os.path.isfile(os.path.join(dir+"/"+name, name_file))]
            if len(files)>70:
                for file in files:
                    data = load_image(dir+"/"+name + "/" + file)
                    X.append(data)
                    Y.append(i)
                i+=1
                names.append(name)
    return X,Y, names

def pca_X(X_train,X_test,h,w):
    n_components = 150
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return eigenfaces,X_train_pca,X_test_pca

def svc_fit(X_train_pca,y_train):
    clf = SVC(kernel='rbf')
    clf = clf.fit(X_train_pca, y_train)
    return clf

def plot_eigenfaces(images, h, w, n_row=3, n_col=3):
    for i in range(n_row * n_col +1):
        plt.subplot(n_row+1, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())

dir = "lfw_funneled"
X,y,names = get_70_faces_dir(dir)
X_flat = [x.ravel() for x in X]
n_features = X_flat[0].size
n_samples = len(X_flat)
h = X[0].shape[0]
w = X[0].shape[1]
n_classes =len(names)
print("Number of samples")
print(n_samples)
print("Number of features")
print(n_features)
print("Number of classes")
print(n_classes)
print("Image shapes")
print(h,w)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.05, random_state=42)


eigenfaces,X_train_pca,X_test_pca = pca_X(X_train,X_test,h,w)
clf = svc_fit(X_train_pca,y_train)
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
plot_eigenfaces(eigenfaces, h, w)

plt.show()
