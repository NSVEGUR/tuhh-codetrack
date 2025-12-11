"""Implementation of the Heaviside function as activation function"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heaviside_like.heaviside"
__package__ = "amllib.activations.heaviside_like"

import numpy as np

from .__base import HeavisideLike


class Heaviside(HeavisideLike):
    """
    Class representation of the Heaviside activation function.

    This class represents the Heaviside activation function
    $$
    \\text{H}(x) = \\begin{cases} \\phantom{-}1, \\quad & x \\geq 0 \\\\ -1, \\quad & x < 0 \\end{cases}.
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
    """

    def __init__(self):
        """
        Initialize the Heaviside activation function.
        """
        super().__init__()
        self.name = 'Heaviside'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Heaviside function.

        This method applies the Heaviside function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary dimension and shape.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """
        return (x >= 0) * 1.0

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the drivative of the Heaviside function.

        This method applies the derivative of the Heaviside function
        componentwise to an array.

        **Note**: Since the Heaviside function has one "jump" at `x = 0`
        the function is not differentiable. Therefore a weak derivative
        is used here, where the point `0` is evaluated to 0.0.

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
        Apply the Heaviside function.

        This method applies the Heaviside function
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

        return (x >= 0) * 1.0

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the drivative of the Heaviside function and multiply the
        result with the input.

        This method applies the derivative of the Heaviside function
        componentwise to the last input of the `feedforward` method.
        The result is then multiplied with the input.

        **Note**: Since the Heaviside function has one "jump" at `x = 0`
        the function is not differentiable. Therefore a weak derivative
        is used here, where the point `0` is evaluated to 0.0.

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
