"""Implementation of the Swish activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.swish"
__package__ = "amllib.activations.relu_like"

import numpy as np

from .__base import ReLULike


class Swish(ReLULike):
    """
    Class representation of the Swish activation function.

    This class represents the Swish activation function
    $$
    \\text{Swish}_k(x) = \\frac{x}{1 + e^{-kx}}
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor : float
        Scaling factor for weight initialization. This factor is shared
        by all ReLU like activation functions. It is set to $\\sqrt{2}$.
    k : float
        Scaling factor for inputs.
    """

    def __init__(self, k: float = 1.0):
        """
        Initialize the Swish activation function.

        Parameters
        ----------
        k : float
            Scaling factor for inputs, by default 1.0
        """

        super().__init__()
        self.name = 'Swish'
        self.k = k
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Swish activation function.

        This method applies the Swish activation function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """

        y = np.exp(-self.k * x)
        return x / (1.0 + y)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Swish activation
        function.

        This method applies the Swish function componentwise
        to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """

        kx = self.k * x
        y = 1.0/(1.0 + np.exp(-kx))
        return y * (1.0 + kx * (1.0 - y))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Swish function.

        This method applies the Swish function
        componentwise to an array. Data is cached
        for later backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------

        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        self.data = self.k * x
        y = np.exp(self.data)
        return x * y/(1.0 + y)


    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Swish function and
        multiply the result with the input.

        This method applies the derivative of the Swish
        function componentwise to the last input of the
        `feedforward` method. The result is then multiplied
        with the input.

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

        y = np.exp(self.data)/(1.0 + np.exp(self.data))
        return y * (1.0 + self.data * (1.0 - y)) * delta
