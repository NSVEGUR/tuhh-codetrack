"""TanH activation function class"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.sign_like.tanh"
__package__ = "amllib.activations.sign_like"

import numpy as np

from .__base import SignLike


class TanH(SignLike):
    """
    Class representation of the TanH activation function.

    This class represents the TanH activation function
    $$
    \\text{TanH}_k(x) = \\tanh(kx)
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

    def __init__(self, k: float = 1.0):
        """
        Initialize the TanH activation function.

        Parameters
        ----------
        k : float
            Scaling factor `> 0` for inputs., by default 1.0
        """

        super().__init__()
        self.name = 'TanH'
        self.k = k
        data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the TanH activation function.

        This method applies the TanH activation function
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

        return np.tanh(self.k * x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the TanH activation function.

        This method applies the derivative of the TanH function
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

        return self.k * (1.0 - np.tanh(self.k * x)**2)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the TanH function.

        This method applies the TanH function
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

        self.data = np.tanh(self.k * x)
        return self.data

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the TanH function and
        multiply the result with the input.

        This method applies the derivative of the TanH
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

        return self.k * (1.0 - self.data**2) * delta
