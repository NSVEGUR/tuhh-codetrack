"""Implementation of the Sign function as activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = 1.1

__name__ = "amllib.activations.sign_like.sign"
__package__ = "amllib.activations.sign_like"

import numpy as np

from .__base import SignLike


class Sign(SignLike):
    """
    Class representation of the Sign activation function.

    This class represents the Sign activation function
    $$
    \\text{Sign}(x) = \\begin{cases} 1, \\quad & x > 0 \\\\
        0, \\quad & x = 0\\\\
        -1, \\quad & x < 0 \\end{cases}.
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

    """

    def __init__(self):
        """
        Initialize the Sign activation function.
        """

        super().__init__()
        self.name = 'Sign'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Sign activation function.

        This method applies the sign activation function
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

        return np.sign(x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the sign function.

        This method applies the derivative of the
        Sign function componentwise to an array.

        **Note**: Since the Sign function is not defferentiable
        in `x = 0`, a weak derivative is used here. The point `0` is
        evaluated to 0.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        return 0.0 * x

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Sign function.

        This method applies the Sign function
        componentwise to an array. For backpropagation
        no data is rquired, thus no data is cached.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        return np.sign(x)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the drivative of the Heaviside function and multiply the
        result with the input.

        This method applies the derivative of the Heaviside function
        componentwise to the last input of the `feedforward` method.
        The result is then multiplied with the input.

        **Note**: Since the Sign function is not defferentiable
        in `x = 0`, a weak derivative is used here. The point `0` is
        evaluated to 0.


        Parameters
        ----------
        delta : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `delta`.
        """
        return 0.0 * delta
