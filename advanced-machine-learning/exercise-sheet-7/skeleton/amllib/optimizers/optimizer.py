"""Abstract base class for optimizer."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.optimizers.optimizer"
__package__ = "amllib.optimizers"

from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimizers
    used for neural networks.

    This class provides a base class with
    common methods each optimizer has to implement.
    It also provides a common base type for all optimizers.
    """

    @abstractmethod
    def update(self, data: list[np.ndarray], ddata: list[np.ndarray]) -> None:
        """
        Update parameters.

        Update parameters given in data with the updates given in ddata.

        Parameters
        ----------
        data : list[np.ndarray]
            List of parameters to update.
        ddata : list[np.ndarray]
            List of updates for all parameters from data.
        """
        pass
