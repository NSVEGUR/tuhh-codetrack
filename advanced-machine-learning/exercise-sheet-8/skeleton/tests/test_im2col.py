"""Test for our implementation of a 2D convolutional layer with im2col."""

import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy.signal import convolve2d

from amllib.layers import Conv2D
from amllib.activations import Abs, ReLU
from amllib.optimizers import SGD


class CheckGradients(unittest.TestCase):

    def setUp(self) -> None:

        # structure
        n = self.n = 2
        c = self.c = 3
        h = self.h = 3
        w = self.w = 4
        m = self.m = 2
        fh = self.fh = 1
        fw = self.fw = 2
        zh = self.zh = h - fh + 1
        zw = self.zw = w - fw + 1

        # initial data
        f = self.f = np.random.randn(m, c, fh, fw)
        b = self.b = np.random.randn(m)

        # training data
        x = self.x = np.random.randn(n, c, h, w)
        y = self.y = np.random.randn(n, m, zh, zw)

        # create the test layer
        self.test_layer = Conv2D(input_shape=(c, h, w),
                                 fshape=(m, fh, fw),
                                 afun=Abs(),
                                 eval_method='im2col')
        self.test_layer.f = f
        self.test_layer.b = b

        # forward step using the method evaluate of the layer
        a = self.a = self.test_layer.feedforward(x)
        z = self.z = self.test_layer._Conv2D__z

        # backprop step using the method backprop of the layer
        # collect all important intermediate results
        self.dx2 = self.test_layer.backprop(a - y)
        self.db2 = self.test_layer.db
        self.df2 = self.test_layer.df

        # Directly calculate results that cannot be recovered
        self.delta = np.sign(z) * (a - y)

        self.C = .5 * np.sum((a - y) * (a - y))
        # parameter for numerical differentiation
        self.hh = 1e-7

        def conv2d(x: np.ndarray, f: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            forward step, used for numerical differentiation

            Parameters
            ----------
            x : np.ndarray
                Input array
            f : np.ndarray
                Filter bank
            b : np.ndarray
                Bias

            Returns
            -------
            np.ndarray
                2D convolution applied to the 4D input array.
            """
            z = np.zeros_like(self.y)
            for i in range(self.n):
                for j in range(self.m):
                    z[i, j, :, :] = b[j]
                    for k in range(self.c):
                        z[i, j, :, :] += convolve2d(x[i, k, :, :],
                                                    f[j, k, :, :],
                                                    mode='valid')
            return z

        self.conv2d = conv2d

    def test_gradient_z(self):

        hh = self.hh
        z = self.z
        y = self.y

        # numerical differentiation for z
        dz = np.zeros_like(z)
        for i in range(self.n):
            for j in range(self.m):
                for k in range(self.zh):
                    for l in range(self.zw):
                        z_tilde = np.copy(z)
                        z_tilde[i, j, k, l] += hh
                        a = np.abs(z_tilde)
                        dz[i, j, k, l] = (.5 * np.sum((a - y)
                                          * (a - y)) - self.C) / hh

        # difference for z
        diff = np.linalg.norm(dz - self.delta)

        self.assertLessEqual(diff, 1e-5, msg=f'The difference of {diff} between '
                                             'the numerical gradient and the gradient '
                                             'computed by the backpropagation of Conv2D '
                                             f'is too great.It should be less than {1e-5}.')

    def test_gradient_b(self):

        # numerical differentiation for b
        db = np.zeros_like(self.b)
        for i in range(self.m):
            b_tilde = np.copy(self.b)
            b_tilde[i] += self.hh
            z = self.conv2d(self.x, self.f, b_tilde)
            a = np.abs(z)
            db[i] = (.5 * np.sum((a - self.y) * (a - self.y)) - self.C) / self.hh

        # difference for b
        diff = np.linalg.norm(db - self.db2)
        self.assertLessEqual(diff, 1e-5, msg=f'The difference of {diff} between '
                                             'the numerical gradient and the gradient '
                                             'computed by the backpropagation of Conv2D '
                                             f'is too great.It should be less than {1e-5}.')

    def test_gradient_f(self):

        # numerical differentiation for f
        df = np.zeros_like(self.f)
        for i in range(self.m):
            for j in range(self.c):
                for k in range(self.fh):
                    for l in range(self.fw):
                        f_tilde = np.copy(self.f)
                        f_tilde[i, j, k, l] += self.hh
                        z = self.conv2d(self.x, f_tilde, self.b)
                        a = np.abs(z)
                        df[i, j, k, l] = (.5 * np.sum((a - self.y)
                                          * (a - self.y)) - self.C) / self.hh

        # difference for f
        diff = np.linalg.norm(df - self.df2)
        self.assertLessEqual(diff, 1e-5, msg=f'The difference of {diff} between '
                                             'the numerical gradient and the gradient '
                                             'computed by the backpropagation of Conv2D '
                                             f'is too great.It should be less than {1e-5}.')

    def test_gradient_x(self):

        # numerical differentiation for x
        dx = np.zeros_like(self.x)
        for i in range(self.n):
            for j in range(self.c):
                for k in range(self.h):
                    for l in range(self.w):
                        x_tilde = np.copy(self.x)
                        x_tilde[i, j, k, l] += self.hh
                        z = self.conv2d(x_tilde, self.f, self.b)
                        a = np.abs(z)
                        dx[i, j, k, l] = (.5 * np.sum((a - self.y)
                                          * (a - self.y)) - self.C) / self.hh

        # difference for x
        diff = np.linalg.norm(dx - self.dx2)
        self.assertLessEqual(diff, 1e-5, msg=f'The difference of {diff} between '
                                             'the numerical gradient and the gradient '
                                             'computed by the backpropagation of Conv2D '
                                             f'is too great.It should be less than {1e-5}.')


class TestConv2D(unittest.TestCase):

    def setUp(self):

        # structure
        n = self.n = 1
        m = self.m = 1
        c = self.c = 1
        fh = self.fh = 2
        fw = self.fw = 2
        h = self.h = 3
        w = self.w = 3
        zh = self.zh = h - fh + 1
        zw = self.zw = w - fw + 1

        # initial data
        f = np.arange(m*c*fh*fw)
        f = self.f = np.reshape(f, (m, c, fh, fw))
        f = f.astype(np.float64)
        b = np.arange(m)
        b = self.b = b.astype(np.float64)

        # training data
        x = np.arange(n*c*h*w)
        x = np.reshape(x, (n, c, h, w))
        x = self.x = x.astype(np.float64)
        y = np.arange(n*m*zh*zw)
        y = np.reshape(y, (n, m, zh, zw))
        y = self.y = y.astype(np.float64)

        self.test_layer2 = Conv2D(input_shape=(self.c, self.h, self.w),
                                  fshape=(self.m, self.fh, self.fw),
                                  afun=ReLU(), optim=SGD(eta=1),
                                  eval_method='im2col')
        self.test_layer2.f = self.f
        self.test_layer2.b = self.b

    def test_eval(self):

        x_conv = self.test_layer2(self.x)
        x_result = [[[[5, 11], [23, 29]]]]

        assert_array_equal(x_conv, x_result, err_msg='The result of the Conv2D evaluation '
                           f'should be {x_result}, but it is {x_conv}.')

    def test_backprop(self):

        x_conv = self.test_layer2.feedforward(self.x)
        delta_conv = self.test_layer2.backprop(x_conv-self.y)
        delta_result = np.array([[[[15, 40, 20], [68, 130, 52], [21, 26, 0]]]])

        assert_array_equal(delta_conv, delta_result, err_msg='The result of the Conv2D backpropagation '
                                                             f'should be {delta_result}, but it is {delta_conv}.')


if __name__ == '__main__':

    unittest.main(verbosity=2)
