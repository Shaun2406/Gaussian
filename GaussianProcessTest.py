# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 12:22:04 2018

@author: smd118
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close()

# Test data
n = 50
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# Define the kernel function, bi-variate gaussian between all points?
#bi-variate gaussian = exp(-.5/sigma^2*))
def kernel(a, b, param):
    sqdist = a**2 + (b**2).T - 2*np.dot(a, b.T) #n x n grid of squared distances
    return np.exp(-.5 * (1/param) * sqdist) #gaussian with width 'param' based on this

param = 0.2
K_ss = kernel(Xtest, Xtest, param)
# Get cholesky decomposition (square root) of the
# covariance matrix
L_ss = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.matmul(L_ss, np.random.normal(size=(n,3)))

# Now let's plot the 3 sampled functions.
#plt.plot(Xtest, f_prior)
#plt.axis([-5, 5, -3, 3])
#plt.title('Three samples from the GP prior')
#plt.show()

# Noiseless training data
Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
ytrain = np.sin(Xtrain)
yreal = np.sin(Xtest)
# Apply the kernel function to our training points
K = kernel(Xtrain, Xtrain, param)
L = np.linalg.cholesky(K + 1e-15*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_s = kernel(Xtrain, Xtest, param)
Lk = np.linalg.solve(L, K_s)
mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# Compute the standard deviation so we can plot it
s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L_s2 = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L_s2, np.random.normal(size=(n,3)))

plt.plot(Xtrain, ytrain, 'bs', ms=8)
plt.plot(Xtest, f_post)
plt.plot(Xtest,yreal, 'k--')
plt.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
plt.plot(Xtest, mu, 'r--', lw=2)
plt.axis([-5, 5, -3, 3])
plt.title('Three samples from the GP posterior')
plt.show()