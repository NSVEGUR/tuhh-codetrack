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