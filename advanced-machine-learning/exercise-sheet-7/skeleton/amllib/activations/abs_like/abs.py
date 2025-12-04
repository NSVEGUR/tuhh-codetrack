"""Implementation of the Abs activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.abs"
__package__ = "amllib.activations.abs_like"

import numpy as np

from .__base import AbsLike


class Abs(AbsLike):
    """
    Class representation of the Abs activation function.

    This class represents the Abs activation function defined as
    $$
    \\text{Abs}(x) = |x|.
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
    """

    def __init__(self):
        """
        Initialize the Abs activation function.
        """
        super().__init__()
        self.name = 'Abs'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Abs activation function.

        This method applies the (shifted) Abs function
        componentwise to an array.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.

        Examples
        --------
            >>> x = np.array([-3., 0., 3.])
            >>> fun = Abs()
            >>> fun.evaluate(x)
            array([3., 0., 3.])
        """

        return np.abs(x)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Abs function.

        This method applies the derivative of the
        Abs function componentwise to an array.

        **Note**: The Abs function is not differentiable in 0.
        Therefore a weak derivative is used, where the value
        0.0 is assumed in 0.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.

        Examples
        --------
            >>> x = np.array([-3., 0., 3.])
            >>> fun = Abs()
            >>> fun.derive(x)
            array([-1., 0., 1.])
        """

        return np.sign(x)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation function and cache data.

        This method applies the Abs function componentwise
        to an array. Data is cached for later backpropagation.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input 'x'.
        """

        self.data = x

        return np.abs(x)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the Abs function.

        This method applies the derivative of the
        Abs function componentwise to the last input
        of the `feedforward` method.
        The result is then multiplied with the input of
        this method.

        **Note**: The Abs function is not differentiable in 0.
        Therefore a weak derivative is used, where the value 0.0
        is assumed in 0.

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

        return np.sign(self.data) * delta
