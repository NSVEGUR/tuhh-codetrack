"""Implementation of the AbsAlpha activation function."""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.abs_like.abs"
__package__ = "amllib.activations.abs_like"

import numpy as np

from .__base import AbsLike


class AbsAlpha(AbsLike):
    """
    Class representation of the AbsAlpha activation function.

    This class represents the (shifted) AbsAlpha activation function.
    For a shifting parameter $\\alpha$ the function is defined as
    $$
    \\text{Abs}_{\\alpha}(x) = \\sqrt{x^2 + \\alpha^2} - \\alpha.
    $$
    For $\\alpha = 0$ it is
    $$
    \\text{Abs}_0(x) = \\text{Abs}(x) = |x|.
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
    alpha : float
            Shifting factor.
    """

    def __init__(self, alpha=1.0):
        """
        Initialize the AbsAlpha activation function.

        Parameters
        ----------
        alpha : float
            Shifting factor, by default 0.0
        """
        super().__init__()
        self.name = 'AbsAlpha'
        self.alpha = alpha
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the AbsAlpha activation function.

        This method applies the (shifted) AbsAlpha function
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
            >>> fun = AbsAlpha(4.0)
            >>> x = np.array([-3., 0., 3.])
            >>> fun.evaluate(x)
            array([1., 0., 1.])
            >>> fun = AbsAlpha()
            >>> fun.evaluate(x)
            array([3., 0., 3.])
            >>> fun = AbsAlpha(-4.0)
            >>> fun.evaluate(x)
            array([9., 8., 9.])
        """

        if (self.alpha == 0.0):
            return np.abs(x)
        else:
            return np.sqrt(x * x + self.alpha**2) - self.alpha

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the AbsAlpha function.

        This method applies the derivative of the
        AbsAlpha function componentwise to an array.

        **Note**: If the shifting parameter is $\\alpha = 0$,
        the AbsAlpha function is not differentiable in 0. Therefore
        a weak derivative is used, where the value 0.0 is assumed
        in 0.

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
            >>> fun = AbsAlpha(4.0)
            >>> x = np.array([-3., 0., 3.])
            >>> fun.derive(x)
            array([-.6, .0, .6])
            >>> fun = AbsAlpha()
            >>> fun.derive(x)
            array([-1., 0., 1.])
            >>> fun = AbsAlpha(-4.0)
            >>> fun.derive(x)
            array([-.6, .0, .6])
        """

        if (self.alpha == 0.0):
            return np.sign(x)
        else:
            return x / np.sqrt(x**2 + self.alpha**2)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the activation function and cache data.

        This method applies the (shifted) AbsAlpha function
        componentwise to an array. Data is cached for
        later backpropagation.

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

        if (self.alpha == 0.0):
            return np.abs(x)
        else:
            return np.sqrt(x*x + self.alpha**2) - self.alpha

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply the derivative of the AbsAlpha function.

        This method applies the derivative of the
        AbsAlpha function componentwise to the last input
        of the `feedforward` method.
        The result is then multiplied with the input of
        this method.

        **Note**: If the shifting parameter is $\\alpha = 0$,
        the AbsAlpha function is not differentiable in 0. Therefore
        a weak derivative is used, where the value 0.0 is assumed
        in 0.

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

        if(self.alpha == 0.0):
            return np.sign(self.data) * delta
        else:
            return self.data/np.sqrt(self.data**2 + self.alpha**2) * delta
