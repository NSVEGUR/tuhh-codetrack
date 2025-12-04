"""Implementation of the Adam optimizer."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.optimizers.adam"
__package__ = "amllib.optimizers"

import numpy as np

from .optimizer import Optimizer


class Adam(Optimizer):
    """
    Class representation of the Adam optimizer.

    This class implements the Adam optimizer discussed
    in the lecture (see lecture notes, section 2.5.7).

    Attributes
    ----------
    eta : float
        Learning rate with which data gets updated.
    beta1 : float
        Decay parameter for the first moment with `0 < beta1 < 1`.
    beta2 : float
        Decay parameter for the second moment with `beta1 < beta2 < 1`.
    eps : float
        Shifting parameter to prevent division by zero.
    v : list[np.ndarray]
        First moments for each data to update.
    w : list[np.ndarray]
        Second moments for each data to update.
    k : int
        Step number for correction scaling.
    """

    def __init__(self, eta: float = .001, beta1: float = .9,
                 beta2: float = .999, eps: float = 1e-8) -> None:
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        eta : float
            Learning rate, by default .001
        beta1 : float
            Decay parameter for the first moments, by default .9
        beta2 : float
            Decay parameter for the second moments, by default .999
        eps : float
            Shifting parameter to prevent division by zero, by default 1e-8
        """

        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.w = []
        self.v = []
        self.k = 0

        self.name = 'Adam'

    def update(self, data: list[np.ndarray], ddata: list[np.ndarray]) -> None:
        """
        Update parameters with the Adam optimizer.

        Update the parameters given in data with the updates given in
        ddata using the Adam method.

        Parameters
        ----------
        data : list[np.ndarray]
            List of parameters.
        ddata : list[np.ndarray]
            List of updates for the parameters given in ddata.
        """

        if self.v == []:
            self.v = [np.zeros_like(p) for p in data]
        if self.w == []:
            self.w = [np.zeros_like(p) for p in data]

        self.k += 1
        alpha = self.eta*np.sqrt(1 - self.beta2**self.k) / \
            (1 - self.beta1**self.k)
        for idx, (p, dp) in enumerate(zip(data, ddata)):

            self.v[idx] = self.beta1 * self.v[idx] + (1 - self.beta1) * dp
            self.w[idx] = self.beta2 * self.w[idx] + (1 - self.beta2) * dp**2

            p -= self.v[idx] * alpha/np.sqrt(self.w[idx] + self.eps)
