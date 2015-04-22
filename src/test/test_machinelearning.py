'''
Created on Apr 21, 2015

@author: roel
'''

import unittest
import numpy as np
from math import tan, cos

import src.base as base
import src.logisticregression as logisticregression
import src.neuralnetwork as neuralnetwork

tan = np.vectorize(tan)
cos = np.vectorize(cos)

npround = lambda x: np.round(x, 5)


class TestBase(unittest.TestCase):


    def test_sigmoid(self):
        one_value = npround(base.sigmoid(1))
        np_array = npround(base.sigmoid(np.array([2, 3])))
        result_array = np.array([0.88080, 0.95257])
        multidim_array = npround(base.sigmoid(base.ml2nparray('[-2 0; 4 999999; -1 1]')))
        result_array_multidim = base.ml2nparray('[0.11920, 0.50000; 0.98201, 1.00000; 0.26894, 0.73106]')
        self.assertEqual(one_value, 0.73106, "base.sigmoid fails with one value")
        self.assertTrue((np_array == result_array).all(), "base.sigmoid fails with multiple values in a numpyarray")
        self.assertTrue((multidim_array == result_array_multidim).all(), "base.sigmoid fails with multiple values in a multidimensional numpyarray")

    def test_add_intercept(self):
        testcase = np.array([[2], [4]])
        testresult = base.add_intercept(testcase)
        testcontrol = np.array([[1, 2], [1, 4]])
        self.assertTrue((testresult == testcontrol).all())


class TestLogisticregression(unittest.TestCase):

    def cf(self, theta, X, y, lmbda, J_result, grad_result, message=''):
        J, grad = logisticregression.costfunction(theta, X, y, lmbda)
        self.assertEqual(int(J[0]*10000)/10000, J_result, message + " the cost is not calculated correctly")
        self.assertTrue((npround(grad) == grad_result).all(), message + " the gradient is not calculated correctly")

    def test_predict(self):
        X, theta = base.ml2nparray('[0.3 ; 0.2]', '[1 2.4; 1 -17; 1 0.5]')
        testresult = logisticregression.predict(X, theta)
        result = np.array([[1], [0], [1]])
        self.assertTrue((testresult == result).all(), 'predict function not working properly')

    def test_costfunction_no_lambda(self):
        theta, X, y  = base.ml2nparray('[-1 ; 0.2]', '[1 34 ; 1 35]', '[0 ; 1]')
        lmbda = 0
        message = "Costfunction without lambda doesn't work properly"
        self.cf(theta, X, y, lmbda, J_result=2.9027, grad_result=np.array([0.49725,16.90542]), message=message)

    def test_costfunction_no_lambda2(self):
        X = np.array([[-0.2751633,  0.424179 ,  0.5403023, -0.1455   , -0.7596879],
                      [-0.532833 ,  0.2836622,  0.7539023,  0.1367372, -0.9576595],
                      [-0.6536436,  0.9601703,  0.9074468,  0.4080821, -0.9999608],
                      [-0.8390715,  0.843854 ,  0.9887046, -0.5477293, -0.9899925],
                      [ 0.0044257,  0.6603167,  0.9912028, -0.4161468, -0.9111303]])
        theta, y = base.ml2nparray('[1;2;3;4;5]', '[1;0;1;0;1]')
        theta = tan(theta)
        lmbda = 0
        message = "Costfunction without lambda doesn't work properly"
        self.cf(theta, X, y, lmbda, J_result=0.6991, grad_result=np.array([-0.09119, -0.01649, 0.05485, -0.01447, -0.07573]), message=message)
        
    def test_costfunction_with_lambda(self):
        theta, X, y,  = base.ml2nparray('[-1 ; 0.2]', '[1 34 ; 1 35]', '[0 ; 1]')
        lmbda = 1.4
        message = "Regularized costfunction with lambda doesn't work properly"
        self.cf(theta, X, y, lmbda, J_result=2.9167, grad_result=np.array([0.49725, 17.04542]), message=message)
        

class TestNeuralnetwork(unittest.TestCase):


    def test_sigmoid_gradient(self):
        a1 = npround(neuralnetwork.sigmoid_gradient(np.array([2, 3])))
        r1 = np.array([0.10499, 0.04518])
        a2 = npround(neuralnetwork.sigmoid_gradient([[-2, 0], [4, 999999], [-1, 1]]))
        r2 = np.array([[0.10499, 0.25000], [0.01766, 0.0], [0.19661, 0.19661]])
        itr = [(a1, r1), (a2, r2)]
        for answer, sought_result in itr:
            self.assertTrue((answer == sought_result).all())
    
    def test_regularized_costfunction(self):
        nn_params = np.array([[x/10 for x in range(1,19)]])
        X = cos(base.ml2nparray('[1 2 ; 3 4 ; 5 6]'))
        y = base.ml2nparray('[4; 2; 3]')
        J, _ = neuralnetwork.costfunction(nn_params=nn_params, layer_sizes=[2], X=X, y=y, 
                                            lmbda=3, input_layer_size=2, num_labels=4)
        self.assertEqual(int(J*10000)/10000, 16.457)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()