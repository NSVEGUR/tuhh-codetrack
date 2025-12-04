"""Test our implementation of a flattening layer."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from amllib.layers import Flatten


class TestFlatten(unittest.TestCase):

    def setUp(self):

        # structure
        self.a = np.arange(16).reshape(2, 2, 2, 2)
        self.test_flattenlayer = Flatten()
        self.y = np.zeros((2, 8))

    def test_eval(self):

        a_flatten = self.test_flattenlayer(self.a)
        a_result = np.arange(16).reshape(2, 8)
        assert_array_equal(a_flatten, a_result, err_msg='The result of the Flatten evaluation '
                           f'should be {a_result}, but it is {a_flatten}.')

    def test_backprop(self):

        y = np.arange(16).reshape(2, 8)
        a_flatten = self.test_flattenlayer.feedforward(self.a)
        delta = a_flatten - self.y
        delta_flatten = self.test_flattenlayer.backprop(delta)
        delta_result = np.arange(16).reshape(2, 2, 2, 2)

        assert_array_equal(delta_flatten, delta_result, err_msg='The result of the Flatten backpropagation '
                                                                f'should be {delta_result}, but it is {delta_flatten}.')


if __name__ == '__main__':

    unittest.main(verbosity=2)
