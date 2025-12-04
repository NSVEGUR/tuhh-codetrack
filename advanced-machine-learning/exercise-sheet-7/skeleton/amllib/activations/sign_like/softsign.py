"""Implementation of the SoftSign activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.sign_like.softsign"
__package__ = "amllib.activations.sign_like"

import numpy as np

from .__base import SignLike


class SoftSign(SignLike):
    """
    Class representation of the SoftSign activation function.

    This class represents the SoftSign activation function
    $$
    \\text{SoftSign}_k(x) = \\frac{kx}{1 + |kx|}
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all Sign like activation functions. It is set to `1.0`.
    k: float
        Scaling factor `> 0` for inputs.
    """

    def __init__(self, k=1.0):
        """
        Initialize the SoftSign activation function.

        Parameters
        ----------
        k : float
            Scaling factor `> 0` for inputs, by default 1.0.
        """

        super().__init__()
        self.name = 'SoftSign'
        self.k = k
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftSign activation function.

        This method applies the SoftSign activation function
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

        return (self.k * x)/(1.0 + np.abs(self.k * x))

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftSign activation function.

        This method applies the derivative of the SoftSign
        activation function componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        return self.k/(1.0 + np.abs(self.k * x))**2

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftSign function.

        This method applies the SoftSign function
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

        self.data = 1.0 + np.abs(self.k * x)
        return self.k * x / self.data

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftSign function and
        multiply the result with the input.

        This method applies the derivative of the SoftSign
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

        return self.k / self.data**2
