"""Implementation of the Linear activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.linear"
__package__ = "amllib.activations.relu_like"

import numpy as np

from .__base import ReLULike


class Linear(ReLULike):
    """
    Class representation of the linear activation function.

    This class represents the linear activation function
    $$
    \\text{f}(x) = x.
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all ReLU like activation functions. It is set to $\\sqrt{2}$.
    """

    def __init__(self):
        """
        Initialize the linear activation function.
        """
        super().__init__()
        self.name = 'Linear'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ReLU activation function.

        This method applies the linear activation function
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

        return x

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the linear activation function.

        This method applies the derivative of the linear function
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
        return 1.0 + 0.0 * x

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the linear activation function.

        This method applies the linear function
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

        self.data = x
        return x

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the linear function and
        multiply the result with the input.

        This method applies the derivative of the linear
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

        return (self.data * 0.0 + 1.0) * delta
