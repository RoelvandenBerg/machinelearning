'''
Created on 10 apr. 2015

@author: Roel van den Berg; roelvdberg_*at*_gmail_*dot*_com

Port to Python from my Octave code for the Stanford Machine Learning course: https://www.coursera.org/course/ml
'''

from src.base import sigmoid, add_intercept
import numpy as np
from scipy import optimize, eye
from math import log

def sigmoid_gradient(z):
    sg = sigmoid(z)
    return sg * (1 - sg)

def forward_propagation(thetas, X, m=None):
    """
    FORWARD_PROPAGATION makes one forward pass for each training case 
    through the neural net by calculating the outcome for each layer.
    
    forward_propagation(thetas, X, m=None)
    
    where:
    thetas =   list of numpy arrays where each array is the theta 
               for that array
    X =        list of training cases (input layer)
    m =        number of training cases, optional argument 
    """
    # determine size of trainingset and number of layers (including input 
    # and output) and initialize a and z layer outcomes  
    if not m:
        m = len(X)
    l = len(thetas)
    a = [None]
    z = [None]
    ai = add_intercept(X, m)
    a.append(ai)
    
    # walk through every layer, calculate a and z for each layer and return
    # the outcome and intermidiate outcomes for each layer a and z  
    for i in range(1, l + 1):
        theta = thetas[i - 1]
        zi = np.dot(ai, np.transpose(theta))
        ai = sigmoid(add_intercept(zi, m))        
        z.append(zi)
        a.append(ai)
    result = ai[:,1:]
    a.append(result)
    
    return result, z, a


def back_propagation(z, a, y, thetas, m, lmbda):
    max_l = len(thetas) - 1
    dl = a[max_l] - y
    d = [None] * (max_l - 1) + dl
    D = [None] * max_l
    
    for l in range(max_l, 0, -1):
        dl = np.dot(dl, thetas[l][:,1:]) * sigmoid_gradient(z[l])
        d[l] = dl
    
    for l in range(max_l):
        theta = thetas[l]
        theta[:,0] = 0
        D[l] = (np.transpose(d[l+1]) * a[l]) / m + lmbda/m *theta

    return D


def cf_regularize(thetas):
    try:
        theta = thetas[0][:,1:]
        result = sum(sum(np.square(theta)))
        result += cf_regularize(thetas[1:])
    except IndexError:
        result = 0


def unpack_theta(nn_params, layer_sizes):
    def itr(layer_sizes):
        tot = 0
        last_tot = 0
        for i in range(len(layer_sizes)-1):
            tot += layer_sizes[i+1] * (layer_sizes[i] + 1)
            yield last_tot, tot, layer_sizes[i+1], layer_sizes[i] + 1
            last_tot = tot

    return [np.reshape(nn_params[:,i:j], (cols, rows)) for i, j, cols, rows in itr(layer_sizes)]


def pack_theta_grad(D):
    return np.hstack([d.flatten('F') for d in D])


def costfunction(nn_params, layer_sizes, X, y, lmbda=0, 
                    input_layer_size=None, num_labels=None):
    """
    COSTFUNCTION Implements the neural network cost function for a multi 
    layer neural network which performs classification
    (J, grad) = costfunction(nn_params, hidden_layer_size, num_labels, ...
    X, y, lambda) computes the cost and gradient of the neural network. The
    parameters for the neural network are "unrolled" into the vector
    nn_params and need to be converted back into the weight matrices. 
    
    The returned parameter grad is an "unrolled" vector of the
    partial derivatives (thetas) of the neural network.
    """
    # initialize each theta from nn_params and get trainingsset length (m)
    if input_layer_size:
        layer_sizes = [input_layer_size] + layer_sizes
    if num_labels:
        layer_sizes = layer_sizes + [num_labels]
    else:
        num_labels = layer_sizes[-1]
    
    thetas = unpack_theta(nn_params, layer_sizes)
    m = len(X)
    
    # apply forward propagation and initialize y as binary array
    result, z, a = forward_propagation(thetas, X, m)
    Q = eye(num_labels)
    y = Q[y,:]
    
    # Calculate cost J based on forward propagation and the thetagradient
    # grad. 
    J = sum(sum(-y * log(result) - (-y + 1) * log(1 - result)))/m + cf_regularize(thetas) * lmbda/(2*m)
    D = back_propagation(z, a, y, thetas, m, lmbda)
    grad = pack_theta_grad(D)
    
    return J, grad

