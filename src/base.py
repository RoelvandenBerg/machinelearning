'''
Created on 10 apr. 2015

@author: Roel van den Berg; roelvdberg_*at*_gmail_*dot*_com

Port to Python from my Octave code for the Stanford Machine Learning course: https://www.coursera.org/course/ml
'''

from math import exp
import numpy as np

def sigmoid(z):
    return 1/(1 + exp(-z))

sigmoid = np.vectorize(sigmoid)


def add_intercept(X, m=None):
    if all(X[:,0]==1):
        return X
    if not m:
        m = len(X)
    ones = np.ones((m,1))
    return np.hstack([ones, X])


