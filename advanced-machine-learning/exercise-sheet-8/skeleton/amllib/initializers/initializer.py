"""Abstract base class for initializers"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.initializers.initializer"
__package__ = "amllib.initializers"

from abc import ABC, abstractmethod

import numpy as np


class KernelInitializer(ABC):
    """
    Base class for kernel initializers.

    This class defines methods every initializer has to implement.
    Each initializer should inherit from this base class.
    """

    @abstractmethod
    def wfun(self, m: int, n: int) -> np.ndarray:
        """
        Initialize a weight matrix of size m by n.

        Parameters
        ----------
        m : int
            Number of rows.
        n : int
            Number of columns

        Returns
        -------
        np.ndarray
            Weight matrix of shape (m, n).
        """

        pass

    @abstractmethod
    def ffun(self, m: int, c: int, fh: int, fw: int) -> np.ndarray:
        """
        Initialize a filter bank for the NCHW format.

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

        Raises
        ------
        NotImplementedError
            Raised if this method is not Implemented.

        Returns
        -------
        np.ndarray
            Filter bank as array of shape (m, c, fh, fw).
        """

        raise NotImplementedError("This method is not implemented for this initializer.")
