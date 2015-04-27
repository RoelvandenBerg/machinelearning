'''
Created on 10 apr. 2015

@author: Roel van den Berg; roelvdberg_*at*_gmail_*dot*_com

Port to Python from my Octave code for the Stanford Machine Learning course (programming assignments): https://www.coursera.org/course/ml
'''

from src.base import sigmoid, add_intercept
import numpy as np
from scipy import optimize, eye


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
    a = []
    z = [None] # 'z' Starts with None, since the input layer does not have to be calculated, but is given directly.  
    ai = add_intercept(X, m)
    a.append(ai)
    
    # walk through every layer, calculate a and z for each layer and return
    # the outcome and intermediate outcomes for each layer a and z  
    for i in range(l):
        theta = thetas[i]
        zi = np.dot(ai, np.transpose(theta))
        ai = add_intercept(sigmoid(zi), m)
        z.append(zi)
        a.append(ai)
    result = ai[:,1:]
    a.append(result)
    
    return result, z, a


def back_propagation(z, a, y, thetas, m, lmbda):
    max_l = len(a) - 1 # max index of a
    dl = a[max_l] - y # Error between forward propagation result and supposed  outcome  
    d = [None] * (max_l - 2) + [dl]
    D = []
    
    for l in range(max_l - 2, 0, -1): # indexing should start at 2 lower than the max index of a since the in- and output layer are excluded
        dl = np.dot(dl, thetas[l][:,1:]) * sigmoid_gradient(z[l]) 
        d[l-1] = dl
    
    for l in range(max_l - 1):
        t = thetas[l]
        theta = add_intercept(t[:,1:], ones=False) # first thetas are not used for the computation and are thus set to zero
        D.append(np.dot(np.transpose(d[l]), a[l])/ m  + lmbda/m *theta)

    return D


def cf_regularize(thetas):
    try:
        theta = thetas[0][:,1:]
        result = sum(sum(np.square(theta)))
        result += cf_regularize(thetas[1:])
    except (TypeError, IndexError):
        result = 0
    return result


def unpack_theta(nn_params, layer_sizes):
    def itr(l_sizes):
        tot = 0
        last_tot = 0
        for i in range(len(l_sizes)-1):
            tot += l_sizes[i+1] * (l_sizes[i] + 1)
            yield last_tot, tot, l_sizes[i+1], l_sizes[i] + 1
            last_tot = tot

    return [np.reshape(nn_params[i:j], (cols, rows), order='F') for i, j, cols, rows in itr(layer_sizes)]


def pack_theta_grad(D):
    return np.hstack(np.reshape(d, -1, 'F') for d in D)


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
    y = Q[y.flatten().astype(int)-1,:]
    # Calculate cost J based on forward propagation and the theta gradient
    # grad. 
    cost = sum(sum(-y * np.log(result) - (-y + 1) * np.log(1 - result)))/m
    regularization = cf_regularize(thetas) * lmbda/(2*m)
    J = cost + regularization

    D = back_propagation(z, a, y, thetas, m, lmbda)
    grad = pack_theta_grad(D)
    
    return J, grad


def rand_init(layer_sizes, epsilon=0.01):
    epsilon *= 2
    size = sum((layer_sizes[i]+1)*layer_sizes[i+1] for i in range(len(layer_sizes)-1))
    return (np.random.rand(size) - 0.5)* epsilon


def nn_optimize(layer_sizes, X, y, lmbda=0, initial_theta=None, method='CG', epsilon=0.01):
    if initial_theta == None:
        initial_theta = rand_init(layer_sizes, epsilon)

    return optimize.minimize(costfunction, initial_theta, (layer_sizes, X, y, lmbda), method, jac=True)