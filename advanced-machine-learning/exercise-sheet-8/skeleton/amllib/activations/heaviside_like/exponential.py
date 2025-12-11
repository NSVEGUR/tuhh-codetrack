"""Implementation of the Exponential function as activation function"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heaviside_like.exponential"
__package__ = "amllib.activations.heaviside_like"

import numpy as np
from .__base import HeavisideLike


class Exp(HeavisideLike):
    """
    Implementation of the exponential activation function.

    This class implements the exponential function as activation function.
    Note that this function does not work well as activation function.

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all Heaviside like activation functions. It is set to 1.0.
    """

    def __init__(self):
        """
        Initialize the exponentail function as activation function.

        On initialization the weight initialization scaling factor
        is set.
        """
        super().__init__()
        self.name = 'Exp'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the exponential function.

        This method applies the exponential function componentwise
        to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        return np.exp(x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the exponential function.

        This method applies the derivative of the exponential
        function, i.e. the exponential function, componentwise
        to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        return np.exp(x)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the exponential function.

        This method applies the exponential function componentwise
        to an array. Data is cached for later backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        self.data = np.exp(x)

        return self.data

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

        return self.data * delta
