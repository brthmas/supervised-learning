import numpy as np

#Basic Logistic Regression

def sig(x):
    y = 1/(1+np.exp(-x))
    return y

def sigGrad(x):
    yPrime = sig(x)*(1-sig(x))
    return yPrime

def gradDe(fPrime, step, x0, thresh = 1e7):
    while np.linalg.norm(fPrime) > thresh:
        x0 = x0 - step*fPrime(x0)
    return x0

def train(X_train, Y_train, threshold = 1e-2):
    ws = 2*np.random.random((X_train.shape[1],1)) - 1
    iterations = 0
    while np.linalg.norm(np.dot(X_train.T, sigGrad(np.dot(X_train,ws))*(sig(np.dot(X_train,ws)) - Y_train))) > threshold:
        ws += -0.1*np.dot(X_train.T, sigGrad(np.dot(X_train,ws))*(sig(np.dot(X_train,ws)) - Y_train))
        iterations += 1
    print(iterations)
    return ws

def function(X, weights):
    return sig(np.dot(X,weights))

X1s = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 1, 0]])
Y1s = np.array([[0, 1, 1, 0, 0]]).T

ws = train(X1s, Y1s)

print(function(np.array([1, 0, 0]), ws))
