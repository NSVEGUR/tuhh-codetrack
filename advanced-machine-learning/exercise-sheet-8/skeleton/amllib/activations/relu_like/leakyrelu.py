"""Implementation of the leaky ReLU activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.leakyrelu"
__package__ = "amllib.activations.relu_like"

import numpy as np

from .__base import ReLULike


class LeakyReLU(ReLULike):
    """
    Class representation of the leaky ReLU activation function.

    This class represents the leaky ReLU activation function
    $$
    \\text{ReLU}_{\\alpha}(x) = \\begin{cases}x, \\quad
    x \\geq 0 \\\\ \\alpha x, \\quad x < 0 \\end{cases}.
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
    alpha: float
        Scaling factor for negative inputs.
    """

    def __init__(self, alpha: float = .01):
        """
        Initialize the leaky ReLU activation function.

        Parameters
        ----------
        alpha : float
            Scaling factor `> 0` for negative inputs, by default 0.01.
        """

        super().__init__()
        self.name = 'leaky ReLU'
        self.alpha = alpha
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the leaky ReLU activation function.

        This method applies the leaky ReLU activation function
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

        return x.clip(min=0) + self.alpha*x.clip(max=0)

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the leaky ReLU function.

        This method applies the derivative of the leaky ReLU
        function componentwise to an array.

        **Note**: Since for $\\alpha\\neq 1$ the leaky ReLU function
        is not differentiable in `x = 0`, a weak derivative is
        used here. The point `0` is evaluated to `1`.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray:
            Output array, has the same shape as the input `x`.
        """

        return (x >= 0) + (x < 0) * self.alpha

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the leaky ReLU function.

        This method applies the leaky ReLU function
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
        return x.clip(min=0) + self.alpha*x.clip(max=0)

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the leaky ReLU function and
        multiply the result with the input.

        This method applies the derivative of the leaky ReLU
        function componentwise to the last input of the
        `feedforward` method. The result is then multiplied
        with the input.

        **Note**: Since for $\\alpha\\neq 1$ the leaky ReLU function
        is not differentiable in `x = 0`, a weak derivative is
        used here. The point `0` is evaluated to `1`.

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

        return (self.data >= 0) * delta +\
                self.alpha * (self.data < 0) * delta
