"""Implementation of the LOCo activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.loco"
__package__ = "amllib.activations.abs_like"

import numpy as np

from .__base import AbsLike


class LOCo(AbsLike):
    """
    Class representation of the LOCo activation function.

    This class represents the LOCo activation function
    $$
    \\text{LOCo}_k(x) = \\frac{\\ln(\\cosh(kx))}{k}.
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all Abs like activation functions. It is set to 1.0.
    k: float
        Scaling factor for inputs of the function.
    """

    def __init__(self, k=1.0):
        """
        Initialize the LoCo activation function.

        Parameters
        ----------
        k : float
            Scaling factor, by default 1.0
        """
        super().__init__()
        self.name = 'LOCo'
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the LOCo activation function

        This method applies the LOCo activation function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        y = self.k * x
        return np.log(np.cosh(y))/self.k

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the LOCo activation function.

        This method applies the derivative of the LOCo function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        y = self.k * x
        return np.tanh(y)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the LOCo activation function

        This method applies the LOCo activation function
        componentwise to an array. Data is cached for
        later backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        y = self.k * x
        self.data = y

        return np.log(np.cosh(y))/self.k

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the LOCo activation function.

        This method applies the derivative of the LOCo function
        componentwise to the last input of the `feedforward` method.
        The result is then multiplied with the input of this method.

        Parameters
        ----------
        delta : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `delta`.

        Raises
        ------
        ValueError
            Raised if the `feedforward` method was not called before.
        """
        if self.data is None:
            raise ValueError('The feedforward method was not called previously. No data'
                             'for backpropagation available')

        return np.tanh(self.data) * delta
