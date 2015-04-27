'''
Created on 10 apr. 2015

@author: Roel van den Berg; roelvdberg_*at*_gmail_*dot*_com

Port to Python from my Octave code for the Stanford Machine Learning course: https://www.coursera.org/course/ml
'''

from math import exp
import numpy as np
import re

def sigmoid(z):
    return 1/(1 + exp(-z))

sigmoid = np.vectorize(sigmoid)


def add_intercept(X, m=None, ones=True):
    if all(X[:,0]==1):
        return X
    if not m:
        m = len(X)
    if ones:
        ones_or_zeros = np.ones((m,1))
    else:
        ones_or_zeros = np.zeros((m,1))
    return np.hstack([ones_or_zeros, X])


def ml2nparray(*args):
    
    def ml2np(string):
        string = string.strip('[]')
        l = [[float(y) for y in re.split(' |,', x) if y != ''] for x in string.split(';')]
        return np.array(l)
    
    try:
        result = ml2np(*args)
    except TypeError:
        result = []
        for arg in args: result.append(ml2np(arg)) 

    return result