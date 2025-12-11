"""Kernel initialization with normal distribution."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers.randn_average"
__package__ = "amllib.initializers"

import numpy as np
from numpy.random import randn
from .initializer import KernelInitializer


class RandnAverage(KernelInitializer):
    """
    Class for random initialization of weight matrices
    and filter banks with a normal distribution.

    This class provides the initialization of weight matrices
    and filter banks with a normal distribution. The mean is
    always 0, and the standard deviation is based on the shape
    of the weight matrix or the filter bank.
    """

    def wfun(self, m: int, n: int) -> np.ndarray:
        """
        Initialize a random weight matrix.

        A random weight matrix of shape $(m,n)$ is generated
        with a normal distribution. The mean is $0$ and the
        standard deviation is $\\sqrt{(m + n) / 2}$.

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns.

        Returns
        -------
        np.ndarray
            A random weight matrix of shape `(m, n)`.
        """

        average = (m + n) / 2.0
        return randn(m, n) / np.sqrt(average)

    def ffun(self, m: int, c: int, fh: int, fw: int) -> np.ndarray:
        """
        Initialize a random filter bank.

        A random filter bank of shape $(m, c, fh, fw)$ is
        generated with a normal distribution.
        The mean is $0$ and standard deviation is
        $\\sqrt{(m + c) * fh * fw / 2}$.

        Parameters
        ----------
        m : int
            Number of filters in the filter bank
        c : int
            Number of channels.
        fh : int
            Filter height.
        fw : int
            Filter width.

        Returns:
            np.ndarray: A random filter bank with shape
            `(m, c, fh, fw)`.
        """

        fan_in = c * fh * fw
        fan_out = m * fh * fw
        average = (fan_in + fan_out) / 2.0

        return randn(m, c, fh, fw) / np.sqrt(average)
