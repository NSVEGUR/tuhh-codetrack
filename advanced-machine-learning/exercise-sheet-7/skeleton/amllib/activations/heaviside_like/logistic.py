"""Implementation of the Logistic activation function"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heaviside_like.logistic"
__package__ = "amllib.activations.heaviside_like"

import numpy as np

from .__base import HeavisideLike


class Logistic(HeavisideLike):
    """
    Class representation of the Logistic function.

    This class represents the Logistic activation function
    $$
    \\text{Logistic}_k(x) = \\frac{1}{1 + e^{-kx}}.
    $$

    Attributes
    ----------
    data: np.ndarray
        Cached data from the `feedforward` method.
    name: str
        Name of the activation function.
    factor: float
        Scaling factor for weight initialization. This factor is shared
        by all Heaviside like activation functions.It is set to 1.0.
    k: float
        Scaling factor for inputs.
    """

    def __init__(self, k: float = 1.0) -> None:
        """
        Initialize the Logistic activation function.

        Parameters
        ----------
        k : float
            Scaling factor for inputs, by default 1.0.
        """

        super().__init__()
        self.name = 'Logistic'
        self.k = k
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Logistic function.

        This method applies the Logistic function
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

        return 1.0/(1.0 + np.exp(-self.k * x))

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Logistic function.

        This method applies the derivative of the Logistic
        function componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        y = 1.0/(1.0 + np.exp(-self.k * x))
        return self.k * y * (1.0 - y)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Logistic function.

        This method applies the Logistic function
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

        self.data = 1.0/(1.0 + np.exp(-self.k * x))
        return self.data

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Logistic function and
        multiply the result with the input.

        This method applies the derivative of the Logistic
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

        return self.k * self.data * (1.0 - self.data) * delta
