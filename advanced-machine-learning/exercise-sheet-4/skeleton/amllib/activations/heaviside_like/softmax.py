"""Implementation of the SoftMax function as activation function"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.activations.heaviside_like.softmax"
__package__ = "amllib.activations.heaviside_like"

import numpy as np
from .__base import HeavisideLike

class SoftMax(HeavisideLike):
    """
    Class representation of the SoftMax activation function.

    This class represents the SoftMax activation function
    $$
    \\text{SoftMax}: \\mathbb{R}^n \\rightarrow (0,1)^n; \\mathbf{x} \\mapsto \\begin{pmatrix}e^{x_1} \\\\ \\vdots \\\\ e^{x_n} \\end{pmatrix} / \\sum_{j = 1}^n e^{x_j}
    $$
    It provides a method to apply the SoftMax function to an array.
    Since the derivative of the SoftMax function is not scalar and not
    needed, the corresponding method returns an array filled with ones.
    This is needed for backpropagation.

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
        Initialize the SoftMax activation function.

        On initialization the weight initialization scaling factor
        is set.
        """

        super().__init__()
        self.name = 'SoftMax'
        self.data = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftMax activation function.

        This method applies the SoftMax function to
        an arrary. If the dimension is greater than 1, the
        SoftMax function is applied to the last axis.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        x_exp = np.exp(x)
        scale = x_exp.sum(axis=-1, keepdims=True)
        return x_exp / scale

    def derive(self, x: np.ndarray) -> np.ndarray:
        """
        Fill an array with ones.

        Since the SoftMax function is multivariate, the derivative is not scalar,
        but the SoftMax function is typically used as activation function for the
        output layer together with a crossentropy loss function for backpropagation.
        Therefore we do not need the derivative of the SoftMax function. For
        the implementation of the backpropagation the `derive` function returns `1.0`.


        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array filled with ones, has the same shape as the input `x`.
        """

        return np.ones_like(x)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the SoftMax activation function.

        This method applies the SoftMax function to
        an arrary. If the dimension is greater than 1, the
        SoftMax function is applied to the last axis. Since
        no data is needed for backpropagation, thus no data
        is cached.

        Parameters
        ----------
        x : np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array, has the same shape as the input `x`.
        """

        x_exp = np.exp(x)
        scale = x_exp.sum(axis=-1, keepdims=True)
        return x_exp / scale

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Fill an array with ones.

        Since the SoftMax function is multivariate, the derivative is not scalar,
        but the SoftMax function is typically used as activation function for the
        output layer together with a crossentropy loss function for backpropagation.
        Therefore we do not need the derivative of the SoftMax function. For
        the implementation of the backpropagation this method is the identity.


        Parameters
        ----------
        delta: np.ndarray
            Input array of arbitrary shape and dimension.

        Returns
        -------
        np.ndarray
            Output array.
        """

        return delta
