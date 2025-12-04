"Abstract base class for regularization."


__name__ = "amllib.regularizers.regularizer"
__package__ = "amllib.regularizers"

from abc import ABC, abstractmethod
import numpy as np

class Regularizer(ABC):
    """
    Abstract base class for neural network regularizer.

    This class provides a base class for neural network
    regularizers that change the parameter updates of a
    layer.
    """

    @abstractmethod
    def regularize(self, ddata: np.ndarray, rdata: np.ndarray):
        """
        Regularize update data with the given data.

        Parameters
        ----------
        rdata : np.ndarray
            Used to regularize the data in ddata.
        ddata : np.ndarray
            Parameter update data.
        """
        pass