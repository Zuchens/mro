import numpy as np
import scipy
import scipy.io as sio
import scipy.spatial
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def neighbors(X_train,Y_train,x_test):
    distances = []
    for x_train in X_train:
        distances.append(scipy.spatial.distance.euclidean(x_train, x_test))
    return Y_train[distances.index(min(distances))]

def neighbor_class(X_train,Y_train,x_test):
    distances = []
    for x_train, y_train in zip(X_train,Y_train):
        distances.append([scipy.spatial.distance.euclidean(x_train, x_test),y_train])
    min_0 = min([x[0] for x in distances if x[1]==[0]])
    min_1 = min([x[0] for x in distances if x[1]==[1]])
    return min_0, min_1


def test_score(X_train,Y_train,X_test,Y_test):
    wrong = 0.0
    for x_test, y_test in zip(X_test, Y_test):
        predicted = neighbors(X_train, Y_train, x_test)
        #print("predicted: " + str(predicted) + "---- expected: " + str(y_test[0]))
        if predicted != y_test[0]:
            wrong += 1
    size = Y_test.size
    print str(wrong / size * 100) + "%"
    return wrong / size * 100

def test_average(X_train,Y_train,X_test,Y_test):
    right = 0
    wrong = 0
    for x_test,y_test in zip(X_test,Y_test):
        min_0, min_1 = neighbor_class(X_train, Y_train, x_test)
        if y_test ==[0]:
            right+=min_0
            wrong+=min_1
        else:
            right+=min_1
            wrong+=min_0
    print(str(right/len(X_test)) + "----" + str(wrong/len(Y_test)))
    return right/len(X_test), wrong/len(X_test)

print "multiDimHypercubes"
mat_contents = sio.loadmat('multiDimHypercubes.mat')
X_train =mat_contents['featuresTrain'][0]
Y_train = mat_contents['classesTrain'][0]
X_test = mat_contents['featuresTest'][0]
Y_test = mat_contents['classesTest'][0]
score = []
for i in range(0,mat_contents['maxDim'][0][0]):
    score.append(test_score(X_train[i], Y_train[i], X_test[i], Y_test[i]))
plt.plot(score)
plt.show()
rights = []
wrongs= []
for i in range(0,mat_contents['maxDim'][0][0]):
    right,wrong = test_average(X_train[i], Y_train[i], X_test[i], Y_test[i])
    rights.append(right)
    wrongs.append(wrong)
x_ses = [i for i in range(0,mat_contents['maxDim'][0][0])]
plt.plot(x_ses, rights,'r.', x_ses,wrongs, 'g.')
plt.show()

