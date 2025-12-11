"""Implementation of the Twist activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.twist"
__package__ = "amllib.activations.abs_like"

import numpy as np

from .__base import AbsLike


class Twist(AbsLike):
    """
    Class representation of the Twist activation function.

    This class represents the Twist activation function
    $$
    \\text{Twist}_k(x) = x\\cdot\\tanh(x)
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
        Initialize the Twist activation function.

        Parameters
        ----------
        k : float
            Scaling factor, by default 1.0
        """
        super().__init__()
        self.name = 'Twist'
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Twist activation function.

        This method applies the Twist activation function
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
        return x * np.tanh(y)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Twist activation
        function.

        This  method applies the derivative of the
        Twist function componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        kx = self.k * x
        y = np.tanh(kx)
        return y + kx * (1.0 - y * y)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Twist activation function.

        This method applies the Twist activation function
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
        kx = self.k * x
        y = np.tanh(kx)
        self.data = [kx, y]

        return x * y

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Twist activation function and
        multiply the result with the input.

        This method applies the derivative of the Twist function
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

        kx, y = self.data
        return (y + kx * (1.0 - y * y)) * delta
