"""Kernel initialization with a uniform distribution."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers.rand_average"
__package__ = "amllib.initializers"

import numpy as np
from numpy.random import rand

from .initializer import KernelInitializer


class RandAverage(KernelInitializer):
    """
    Class for random initialization of weight matrices
    and filter banks with a uniform distribution.

    This class provides the initialization of weight matrices
    and filter banks with uniform distribution on an interval
    $[-r,r]$, where the radius r depends on the shape of the
    weight matrix or the filter bank.
    """

    def wfun(self, m: int, n: int) -> np.ndarray:
        """
        Initialize a random weight matrix.

        A random weight matrix of shape $(m, n)$ is generated
        with uniform distribution on the interval $[-r,r]$,
        where $r = \\sqrt{6 / (m + n)}$

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns.

        Returns
        -------
        np.ndarray
            A weight matrix of shape `(m, n)`
        """

        average = (m + n) / 2.0
        return (rand(m, n) - .5) * np.sqrt(12.0 / average)

    def ffun(self, m: int, c: int, fh: int, fw: int) -> np.ndarray:
        """
        Initialize a random filter bank of shape $(m, c, fh, fw)$
        with uniform distribution on the interval $[-r, r]$, where
        $r = \\sqrt{6 / ((m + c) * fh * fw)}$

        Parameters
        ----------
        m : int
            Number of filters in the filter bank.
        c : int
            Number of channels.
        fh : int
            Filter height.
        fw : int
            Filter width.

        Returns
        -------
        np.ndarray
            A random filter bank of shape (m, c, fh, fw).
        """

        fan_in = c * fh * fw
        fan_out = m * fh * fw
        average = (fan_in + fan_out) / 2.0

        return (rand(m, c, fh, fw) - .5) * np.sqrt(12.0 / average)
