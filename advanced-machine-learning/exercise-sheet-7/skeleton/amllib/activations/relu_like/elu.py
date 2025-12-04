"""Implementation of the ELU activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.relu_like.elu"
__package__ = "amllib.activations.relu_like"

import numpy as np

from .__base import ReLULike


class ELU(ReLULike):
    """
    Class representation of the ELU activation function.

    This class represents the ELU activation function
    $$
    \\text{ELU}_{\\alpha}(x) = \\begin{cases}x, \\quad
    &x \\geq 0 \\\\ (\\exp(x) - 1)\\alpha, \\quad &x < 0 \\end{cases}.
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

    def __init__(self, alpha: float = 1.0):
        """
        Initialize the ELU activation function.

        Parameters
        ----------
        alpha : float
            Scaling factor for negative inputs, by default 1.0.
        """

        super().__init__()
        self.name = 'ELU'
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ELU activation function.

        This method applies the ELU activation function
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

        return x.clip(min=0) + \
            (np.exp(x.clip(max=0)) - 1.0) * self.alpha

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the ELU activation function.

        This method applies the derivative of the ELU function
        componentwise to an array.

        **Note**: Since for $\\alpha\\neq 1$ the ELU function
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

        y = np.exp(x.clip(max=0))
        return (x >= 0) + (x < 0) * self.alpha * y

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the ELU activation function.

        This method applies the ELU activation function
        componentwise to an array. Data is cached for
        later backpropagation.

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
        return x.clip(min=0) + \
              (np.exp(x.clip(max=0))- 1.0) * self.alpha

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the ELU function and
        multiply the result with the input.

        This method applies the derivative of the ELU
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

        y = (self.data >= 0) + \
            (self.data < 0) * self.alpha * np.exp(self.data)
        return y * delta
