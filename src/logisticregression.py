'''
Created on 9 apr. 2015

@author: Roel van den Berg; roelvdberg_*at*_gmail_*dot*_com

Port to Python from my Octave code for the Stanford Machine Learning course  (programming assignments): https://www.coursera.org/course/ml
'''

from src.base import sigmoid #, add_intercept
import numpy as np
from scipy import optimize


def predict(theta, X):
    return np.dot(X, theta) > 0

def costfunction(theta, X, y, lmbda=0):
    '''
    COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    J = COSTFUNCTIONREG(theta, X, y, lmbda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    ''' 
    m = len(y)
    thetaslice = np.append(0, theta[1:])
    Xtheta = sigmoid(np.dot(X, theta))
    J = sum(-y * np.log(Xtheta) - (1 - y) * np.log(1 - Xtheta))/m + sum(np.square(thetaslice)) * lmbda / (2 * m)
    grad = sum((Xtheta - y) * X) / m + thetaslice * lmbda/m
    return J, grad

def oneVsAll(X, y, num_labels, lmbda):
    pass

def lr_optimize(initial_theta, X, y, lmbda=0, method='CG'):
    return optimize.minimize(costfunction, initial_theta, (X, y, lmbda), method, jac=True)
