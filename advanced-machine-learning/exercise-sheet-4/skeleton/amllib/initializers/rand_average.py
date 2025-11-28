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

        # TODO Implement the initialization of a m by n matrix with
        # a uniform distribution. The interval is given in the docstring above.
        # Use the numpy.random.rand function to obtain a matrix with entries uniformly
        # distributed on the interval [0,1], and scale and shift the entries to
        # obtain the distribution on a different interval.
