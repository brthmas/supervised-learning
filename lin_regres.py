import numpy as np
import matplotlib.pyplot as plt

d0 = -10
d1 = 10

domain = np.linspace(d0, d1, 2000)
ran = np.random.random_sample

def f1(x):
    return x - 2

#generate random points around the line f(x) = x-2
x1s = (d1-d0)*ran(50) + d0
ys = f1(x1s) + (10*ran(50) - 5)

#generalizes to N-dimensional linear regression
#y = x0 + w*x, output of the form (x0, w1, w2, ..., w(N-1))
def linRegres(solution,*tests):
    '''
    generalizes to N-dimensional linear regression
    y = x0 + w*x, output of the form (x0, w1, w2, ..., w(N-1))
    '''
    X = np.column_stack((tuple(tests)))
    A = np.hstack((np.ones((tests[0].size,1)),X))
    weights = np.dot(np.linalg.pinv(A), solution)
    return weights

reg = linRegres(ys,x1s)

print(reg)

#test for the 2-dimensional case
plt.plot(x1s, ys, 'or')
plt.plot(domain, reg[0] + reg[1] * domain)
plt.show()