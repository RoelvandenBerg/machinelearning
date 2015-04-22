'''
Created on Apr 21, 2015

@author: roel
'''
import unittest
import test_neuralnetwork
from .. import base
from .. import logisticregression
from .. import neuralnetwork
import numpy as np

class TestBase(unittest.TestCase):


    def test_sigmoid(self):
        one_value = base.sigmoid(1)
        np_array = base.sigmoid(np.array([2, 3]))
        result_array = np.array([0.88080, 0.95257])
        self.assertEqual(one_value, 0.73106)
        self.assertEqual(np_array, result_array)

    def testName(self):
        pass
        #costfunction(theta, X, y, lmbda=0)
        #[[1, 2], [1, 3], [1, 4], [1, 5]]
        #[7;6;5;4]
        #[0.1;0.2]
        #logisticregression.costfunction( , ,  )
        #ans =  11.9450


class TestLogisticregression(unittest.TestCase):


    def testName(self):
        pass


class TestNeuralnetwork(unittest.TestCase):


    def testName(self):
        pass



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()