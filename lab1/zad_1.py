

import numpy as np
import scipy
import scipy.io as sio
import scipy.spatial
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


def neighbors(X_train,Y_train,x_test):
    distances = []
    for x_train in X_train:
        distances.append(scipy.spatial.distance.euclidean(x_train, x_test))
    return Y_train[distances.index(min(distances))]

def test_score(X_train,Y_train,X_test,Y_test):
    wrong = 0.0
    for x_test, y_test in zip(X_test, Y_test):
        predicted = neighbors(X_train, Y_train, x_test)
        # print("predicted: " + str(predicted) + "---- expected: " + str(y_test[0]))
        if predicted != y_test[0]:
            wrong += 1
    size = Y_test.size
    print str(wrong / size * 100) + "%"
    return

def shuffle_two(list1,list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(list1))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])
    return list1, list2
# print "FACES"
# mat_contents = sio.loadmat('facesYale.mat')
# X_train =mat_contents['featuresTrain']
# Y_train = [x[0] for x in mat_contents['personTrain']]
# X_test = mat_contents['featuresTest']
# Y_test = mat_contents['personTest']
#
# test_score(X_train,Y_train,X_test,Y_test)

# for i in range(1,20):
#     X_train, Y_train = shuffle_two(X_train, Y_train)
#     X_train = mat_contents['featuresTrain']
#     X_train[:,9] *= i
#     X_test = mat_contents['featuresTest']
#     X_test[:,9] *= i
#     test_score(X_train, Y_train, X_test, Y_test)


print("SPAMBASE")
mat_contents = sio.loadmat('spambase.mat')
X_train =mat_contents['featuresTrain']
Y_train = [x[0] for x in mat_contents['classesTrain']]
X_test = mat_contents['featuresTest']
Y_test = mat_contents['classesTest']
test_score(X_train,Y_train,X_test,Y_test)

for i in range(0,10):
    X = np.append(mat_contents['featuresTrain'], mat_contents['featuresTest'], axis=0)
    Y = np.append(mat_contents['classesTrain'], mat_contents['classesTest'], axis=0)
    # X_train, Y_train = shuffle_two(X, Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    Y_train = [x[0] for x in Y_train]
    test_score(X_train, Y_train, X_test, Y_test)