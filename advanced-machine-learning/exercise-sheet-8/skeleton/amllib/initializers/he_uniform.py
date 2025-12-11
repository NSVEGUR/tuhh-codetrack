"""Kernel initialization proposed by He using a uniform distribution."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers.he_uniform"
__package__ = "amllib.initializers"


import numpy as np
from numpy.random import rand

from .initializer import KernelInitializer


class HeUniform(KernelInitializer):
    """
    Class for the random initialization of weights
    and filter banks as proposed by He et al.
    (See lecture notes, section 2.3).
    """

    def wfun(self, m: int, n: int) -> np.ndarray:
        """
        Initialize a random weight matrix.

        A random weight matrix of shape $(m, n)$ is generated
        with a uniform distribution on the interval $[-r, r]$,
        where $r = \\sqrt{6 / m}$

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

        return (rand(m, n) - .5) * 2 * np.sqrt(6 / m)

    def ffun(self, m: int, c: int, fh: int, fw: int) -> np.ndarray:
        """
        Initialize a random filter bank.

        A random filter bank of shape $(m, c, fh, fw)$ is generated
        with a uniform distribution on the interval $[-r, r]$, where
        $r = \\sqrt{6 / (c * fh * fw)}$.

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
            A random filter bank of shape `(m, c, fh, fw)`.
        """

        fan_in = c * fh * fw

        return (rand(m, c, fh, fw) - .5) * 2 * np.sqrt(6 / fan_in)
