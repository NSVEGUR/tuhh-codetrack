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

        # TODO Initialize a random matrix with normal distributed entries as
        # stated in the docstring above.
        # Use numpy.random.randn to obtain entries normal distributed with mean 0 and
        # standard deviation 1. Shift the results to obtain a different mean, and scale
        # them to obtain a different standard deviation.
