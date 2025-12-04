"""Abstract base class for loss functions."""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.0'

__name__ = 'amllib.losses.loss'
__package__ = 'amllib.losses'

from abc import ABC

import numpy as np
from numpy.typing import ArrayLike


class Loss(ABC):
    """
    Abstract base class for loss functions.

    This abstract base class defines common methods for
    all loss functions. This includes:
    """

    def __call__(self, network_output: ArrayLike,
                 target_output: ArrayLike) -> float:
        """
        Calculate the (mean) loss for given network outputs and
        target outputs.

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target output.

        Returns
        -------
        float
            (Mean) loss for the given (batch of) network outputs and
            target outputs.
        """
        pass

    def backprop(self, network_output: ArrayLike,
                 target_output: ArrayLike) -> np.ndarray:
        """
        Compute the (mean) derivative of the loss for given network outputs
        and target outputs.

        Parameters
        ----------
        network_output : ArrayLike
            Output of a network which is compared to the target output.
        target_output : ArrayLike
            Target outputs.

        Returns
        -------
        float
            Derivative of the loss for the given (batch of) network
            outputs and target outputs.
        """

        pass
