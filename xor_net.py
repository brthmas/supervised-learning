#Learn XOR on [-1,1]x[-1,1]

import numpy as np
import matplotlib.pyplot as plt

'''
Generate XOR Dataset
'''

def xor(x,y):
    return np.sign(x*y)

Xs = 2*np.random.random((2,100)) - 1
x1,x2 = Xs
Ys = xor(x1,x2)

'''
Define Activation Map
'''

def activ(x, deriv = False):
    y = np.tanh(x)
    if deriv == True:
        return 1-y**2
    else:
        return y

'''
FFNN Architecture
'''

in_dim = Xs.shape[0]
h1_dim = 5
out_dim = 1

#Initialize weights
w1s = 2*np.random.random((h1_dim,in_dim)) - 1
w2s = 2*np.random.random((1,h1_dim)) - 1

#Initialize biases
b1 = (2*np.random.random() - 1)*np.ones((h1_dim, 1))
b2 = (2*np.random.random() - 1)*np.ones((out_dim, 1))

def xorNN(X, weights1 = w1s, weights2 = w2s, bias1 = b1, bias2 = b2):
    l1 = activ(np.dot(weights1,X) + bias1)
    l2 = activ(np.dot(weights2,l1) + bias2)
    return l2

def cost(X,Y, weights1 = w1s, weights2 = w2s, bias1 = b1, bias2 = b2):
    return 0.5*(1/100)*np.linalg.norm(xorNN(X, weights1, weights2, bias1, bias2) - Y)**2

passes = 0
lrate = 0.01

while passes < 100000:
    #forward prop
    l0 = Xs
    l1 = activ(np.dot(w1s,l0) + b1)
    l2 = activ(np.dot(w2s,l1) + b2)

    #back prop
    l2_error = l2 - Ys
    l2_delta = l2_error*activ(w2s.dot(l1) + b2, deriv=True)

    l1_error = w2s.T.dot(l2_delta)
    l1_delta = l1_error*activ(w1s.dot(l0)+ b1, deriv=True)

    dcost_dw2 = l2_delta.dot(l1.T)
    dcost_dw1 = l1_delta.dot(l0.T)

    #update weights
    w2s = w2s - lrate*dcost_dw2
    w1s = w1s - lrate*dcost_dw1

    b2 = b2 - lrate*l2_delta.dot(np.ones((Xs.shape[1],1)))
    b1 = b1 - lrate*l1_delta.dot(np.ones((Xs.shape[1],1)))

    passes += 1
    if (passes % 10000) == 0:
        print(cost(Xs,Ys,w1s,w2s,b1,b2))

X = np.array([[1,1,-1,-1],
              [1,-1,1,-1]])

X_test = 2*np.random.random((2,1000)) - 1

for column in X_test.T:
    x,y = column
    if xorNN(column.reshape((2,1)), w1s, w2s, b1, b2) > 0:
        plt.plot(x,y, 'ro')
    else:
        plt.plot(x,y, 'bo')

plt.show()
