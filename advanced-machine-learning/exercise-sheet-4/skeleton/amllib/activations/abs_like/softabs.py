"""Implementation of the SoftAbs activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.softabs"
__package__ = "amllib.activations.abs_like"

import numpy as np

from .__base import AbsLike


class SoftAbs(AbsLike):
    """
    Class representation of the SoftAbs activation function.

    This class represents the SoftAbs activation function
    $$\\text{SoftAbs}_k(x) = \\frac{kx^2}{1 + |kx|}.$$

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
        Initialize the SoftAbs activation function.

        Parameters
        ----------
        k: float
            Scaling factor, by default 1.0.
        """
        super().__init__()
        self.name = 'SoftAbs'
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftAbs activation function.

        This method applies the SoftAbs activation function
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
        return y * x / (1.0 + np.abs(y))

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftAbs activation
        function.

        This method applies the derivative of the SoftAbs
        function componentwise to an array.

        Parameters
        ----------
        x: np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """
        y = 1.0 + np.abs(self.k * x)
        return (1.0 + y)/(y**2) * self.k * x

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftAbs activation function

        This method applies the SoftAbs activation function
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

        return y * x / (1.0 + np.abs(y))

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftAbs activation function and
        multiply the result with the input.

        This method applies the derivative of the SoftAbs function
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

        y = 1.0 + np.abs(self.data)

        return (1.0 + y)/(y**2) * self.data
