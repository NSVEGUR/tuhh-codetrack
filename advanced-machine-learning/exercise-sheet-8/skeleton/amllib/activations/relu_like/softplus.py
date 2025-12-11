"""Implementation of the SoftPlus activation function"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.softplus"
__package__ = "amllib.activations.relu_like"

import numpy as np

from .__base import ReLULike


class SoftPlus(ReLULike):
    """
    Class representation of the SoftPlus activation function.

    This class represents the SoftPlus activation function
    $$
    \\text{SoftPlus}_k(x) = \\frac{\\ln(e^{kx} + 1)}{k}
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
    k : float
        Scaling factor for inputs.
    """

    def __init__(self, k: float = 1.0):
        """
        Initialize the SoftPlus activation function.

        Parameters
        ----------
        k : float
            Scaling factor for inputs, by default 1.0
        """

        super().__init__()
        self.name = 'SoftPlus'
        self.k = k
        self.data = None

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

        return np.log(np.exp(self.k * x) + 1.0)/self.k

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftPlus activation function.

        This method applies the derivative of the SoftPlus activation
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

        return 1.0 / (1.0 + np.exp(-self.k * x))

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftPlus function.

        This method applies the SoftPlus function
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

        self.data = np.exp(self.k * x)
        return np.log(self.data + 1.0)/self.k

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the SoftPlus function and
        multiply the result with the input.

        This method applies the derivative of the SoftPlus
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

        return self.data / (1.0 + self.data) * delta
