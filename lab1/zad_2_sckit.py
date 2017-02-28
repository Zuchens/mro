import scipy.io as sio
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np




def knn_function(X_train,Y_train,X_test,Y_test):
    Y_train = [x[0] for x in Y_train]
    neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
    neigh.fit(X_train,Y_train)
    wrong = 0.0
    distances  = []
    for person, person_class in zip(X_test,Y_test ):
        predicted = neigh.predict(person.reshape(1, -1))[0]
        distances.append(neigh.kneighbors(person.reshape(1, -1))[0])
        if predicted != person_class[0]:
            wrong += 1
    size = Y_test.size
    print "False: "+ str(wrong / size) + "\t Avg distances: " + str(np.mean(distances))

print "multiDimHypercubes"
mat_contents = sio.loadmat('multiDimHypercubes.mat')
X_train =mat_contents['featuresTrain'][0]
Y_train = mat_contents['classesTrain'][0]
X_test = mat_contents['featuresTest'][0]
Y_test = mat_contents['classesTest'][0]
for i in range(0,mat_contents['maxDim'][0][0]):
    knn_function(X_train[i], Y_train[i], X_test[i], Y_test[i])

