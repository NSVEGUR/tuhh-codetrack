"""Implementation of the SGD optimizer."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.optimizers.sgd"
__package__ = "amllib.optimizers"

import numpy as np

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Class representation of the SGD optimizer
    for neural networks.

    This class implements the SGD optimizer
    (with out without momentum) for the training
    of neural networks.

    Attributes
    ----------
    eta : float
        Learning rate for the updates.
    momentum : float
        Decay parameter for the first moments.
    v : list[np.ndarray]
        List of first moments for the data to update.
    """

    def __init__(self, eta: float = .01, momentum: float = 0.0) -> None:
        """
        Initialize the SGD optimizer.

        Parameters
        ----------
        eta : float
            Learning rate, by default .01
        momentum : float, optional
            Decay parameter for the first moments, by default 0.0
        """

        self.eta = eta
        self.momentum = momentum
        if self.momentum > 0.0:
            self.v = []
            self.name = f'SGD({eta}) with momentum {momentum}'
        else:
            self.name = f'SGD({eta})'

    def update(self, data: list[np.ndarray], ddata: list[np.ndarray]) -> None:
        """
        Update parameters with the SGD method.

        Update the parameters given in data with the updates given
        in ddata using the SGD method.

        Parameters
        ----------
        data : list[np.ndarray]
            List of parameters to update.
        ddata : list[np.ndarray]
            List of updates for the parameters given in ddata.
        """
        if self.momentum > 0.0:

            if self.v == []:
                self.v = [np.zeros_like(p) for p in data]

            for idx, (p, dp) in enumerate(zip(data, ddata)):

                self.v[idx] = v * self.momentum + dp
                p -= self.v[idx] * self.eta

        else:

            for p, dp in zip(data, ddata):

                p -= dp * self.eta
