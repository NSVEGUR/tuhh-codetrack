"""Test our implementation of a max-pooling layer."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from amllib.layers import MaxPool2D


class TestMaxPool2D(unittest.TestCase):

    def setUp(self):

        self.a = np.arange(16).reshape(2, 2, 2, 2)
        self.area = (2, 2)
        self.test_poolinglayer = MaxPool2D(self.area)
        self.y = np.arange(2*2).reshape(2, 2, 1, 1)

    def test_eval(self):

        a_pooling = self.test_poolinglayer(self.a)
        a_result = np.array([[[[3]], [[7]]], [[[11]], [[15]]]])

        assert_array_equal(a_pooling, a_result, err_msg='The result of the MaxPool2D evaluation '
                           f'should be {a_result}, but it is {a_pooling}.')

    def test_backprop(self):

        a_pooling = self.test_poolinglayer.feedforward(self.a)
        delta = a_pooling - self.y
        delta_pooling = self.test_poolinglayer.backprop(delta)

        # Compare with the correct result:

        delta_result = np.array([[[[0, 0], [0, 3]], [[0, 0], [0, 6]]], [
            [[0, 0], [0, 9]], [[0, 0], [0, 12]]]])

        assert_array_equal(delta_pooling, delta_result, err_msg='The result of the MaxPool2D backpropagation '
                                                                f'should be {delta_result}, but it is {delta_pooling}.')

    def test_strict(self):

        self.test_poolinglayer.strict = True
        self.a[0, 0, 0, 0] = 3
        self.a[0, 0, 0, 1] = 3
        self.a[0, 0, 1, 0] = 3
        a_pooling = self.test_poolinglayer.feedforward(self.a)
        delta = delta = a_pooling - self.y
        delta_pooling = self.test_poolinglayer.backprop(delta)

        # Compare with the correct result:

        delta_result = np.array([[[[0.75, 0.75], [0.75, 0.75]], [[0, 0], [0, 6]]], [
            [[0, 0], [0, 9]], [[0, 0], [0, 12]]]])

        assert_array_equal(delta_pooling, delta_result, err_msg='The result of the MaxPool2D backpropagation '
                                                                f'should be {delta_result}, but it is {delta_pooling}.')


if __name__ == '__main__':

    unittest.main(verbosity=2)
