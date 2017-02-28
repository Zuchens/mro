import scipy.io as sio
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np




def knn_function(X_train,Y_train,X_test,Y_test):
    neigh = KNeighborsClassifier(n_neighbors=1,metric='euclidean')
    neigh.fit(X_train,Y_train)
    wrong = 0.0
    # predicted =  neigh.predict(X_test)
    # print f1_score(Y_test, predicted, average=None)
    for person, person_class in zip(X_test,Y_test ):
        predicted = neigh.predict(person.reshape(1, -1))[0]
        # print "predicted:" + str(predicted) + "\tactual:" + str(person_class[0])
        if predicted != person_class[0]:
            wrong += 1
    size = Y_test.size
    print wrong / size
print "FACES"
mat_contents = sio.loadmat('facesYale.mat')
X_train =mat_contents['featuresTrain']
Y_train = [x[0] for x in mat_contents['personTrain']]
X_test = mat_contents['featuresTest']
Y_test = mat_contents['personTest']
knn_function(X_train,Y_train,X_test,Y_test)
for i in range(1,20):
    X_train = mat_contents['featuresTrain']
    X_train[:,9] *= i
    X_test = mat_contents['featuresTest']
    X_test[:,9] *= i
    knn_function(X_train, Y_train, X_test, Y_test)

print("SPAMBASE")
mat_contents = sio.loadmat('spambase.mat')
X_train =mat_contents['featuresTrain']
Y_train = [x[0] for x in mat_contents['classesTrain']]
X_test = mat_contents['featuresTest']
Y_test = mat_contents['classesTest']
knn_function(X_train, Y_train, X_test, Y_test)

for i in range(0,10):
    X = np.append(mat_contents['featuresTrain'], mat_contents['featuresTest'], axis=0)
    Y = np.append(mat_contents['classesTrain'], mat_contents['classesTest'], axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    Y_train = [x[0] for x in Y_train]
    knn_function(X_train, Y_train, X_test, Y_test)