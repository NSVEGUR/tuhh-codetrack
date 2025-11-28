"""Implementation of a dense connected layer."""

__author__ = 'Jens-Peter M. Zemke, Jonas Grams'
__version__ = '1.1'

__name__ = "amllib.layers.dense"
__package__ = "amllib.layers"

from typing import Optional, Union

import numpy as np

from amllib.initializers import KernelInitializer, RandnAverage
from amllib.activations import Activation, ReLU
from amllib.layers import Layer


class Dense(Layer):
    """
    Class represention of a dense connected layer.

    This class represents dense connected layers. It is derived
    from the abstract base class `Layer`, and additionally provides
    the methods `set_weights` and `set_bias`.

    Attributes
    ----------
    input_shape: int
        Number of inputs per batch. This is attribute is initialized
        when the build method is called.
    output_shape: int
        Number of outputs per batch.
    afun: Activation
        Activation function of the layer.
    learning_rate: float
        Learning rate for the parameter updates.
    initializer: KernelInitializer
        Initializer used to initializer the weights. The initializer
        is called in the build method.
    __z: np.ndarray
        Last computed affine linear combination.
    __a: np.ndarray
        Last computed activation (output) of the layer.
    W : np.ndarray
        Weight matrix of shape `(ni, no)`.
    b : np.ndarray
        Bias of shape `(no,)`.
    dW : np.ndarray
        Storage for the update of the weight matrix
    db : np.ndarray
        Storage for the update of the bias.
    """

    def __init__(self,
                 output_shape: Union[int, tuple[int]],
                 input_shape: Optional[Union[int, tuple[int]]] = None,
                 afun: Optional[Activation] = None,
                 initializer: Optional[KernelInitializer] = None,
                 learning_rate: float = 0.01) -> None:
        """
        Initialize the dense connected layer.

        Parameters
        ----------
        output_shape : Union[int, tuple[int]]
            Number of outputs (neurons of the layer, as 1-tuple or integer).
        input_shape : Union[int, tuple[int]], optional
            Shape of the inputs, by default None. If it is not None,
            the layer is built on initialization.
        afun : Activation
            Activation function, by default ReLU.
        initializer : KernelInitializer
            Initializer for the weights, by default RandnAverage.
        learning_rate : float
            Learning rate for the parameter updates.
        """
        if type(output_shape) is tuple:
            (self.no,) = output_shape
        else:
            self.no = output_shape

        # Set ReLU as default activation function
        if afun is None:
            self.afun = ReLU()
        else:
            self.afun = afun

        # Set RandnAverage as default weight matrix initializer
        if initializer is None:
            self.initializer = RandnAverage()
        else:
            self.initializer = initializer

        self.learning_rate = learning_rate

        self.__z = None
        self.__a = None

        # If an input shape is given, build layer.
        if input_shape is not None:

            self.build(input_shape)

    def build(self, input_shape: Union[int, tuple[int]]) -> None:
        """
        Build the layer.

        All weights and biases are initialized according to
        the input shape.

        Parameters
        ----------
        input_shape : Union[int, tuple[int]]
            Shape of the inputs (without the batch dimension).
        """
        super().__init__()

        if type(input_shape) is tuple:
            (self.ni,) = input_shape
        else:
            self.ni = input_shape

        # Initialize weights and biases
        self.W = self.afun.factor * self.initializer.wfun(self.ni, self.no)
        self.b = np.zeros(self.no)

        # Allocate storage for the updates.
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self.built = True

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer.

        Evaluate the affine linear combination with
        the weight matrix and the bias and activate
        the result.

        Parameters
        ----------
        a : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Activated output array.
        """

        # Build the layer with the shape of the inputs, if not done yet
        if not self.built:
            self.build(a.shape[1:])

        z = a @ self.W + self.b
        return self.afun(z)

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Evaluate the layer.

        Evaluate the affine linear combination with
        the weight matrix and the bias and activate
        the result. Data is cached for backpropagation.

        Parameters
        ----------
        a : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Activated output array.
        """

        # Build the layer with the shape of the inputs, if not done yet
        if not self.built:
            self.build(a.shape[1:])

        # TODO Cache data for backpropagation (inputs and affine linear combinations)
        self.__a = a
        self.__z = self.__a @ self.W + self.b
        # TODO Return the activated output.
        return self.afun.feedforward(self.__z)

    def set_weights(self, W: np.ndarray) -> None:
        """
        Set the weight matrix of the layer.

        Parameters
        ----------
        W : np.ndarray
            New weight matrix.

        Raises
        ------
        ValueError
            Raised if the shape of the new weight matrix does
            not match the weight matrix shape of the layer.
        """

        if(W.shape != (self.ni, self.no)):
            raise ValueError("The new weight matrix has the wrong shape."
                             f"It should have shape {(self.ni, self.no)}"
                             f"but has shape {W.shape}.")
        self.W = W

    def set_bias(self, b: np.ndarray):
        """
        Set the bias of the layer.

        Parameters
        ----------
        b : np.ndarray
            New bias.

        Raises
        ------
        ValueError
            Raised if the size of the new bias does not match
            the bias size of the layer.
        """
        if(b.size != self.no):
            raise ValueError("The new bias has the wrong size."
                             f"It should have size {self.no} but"
                             f"has size {b.size}.")
        self.b = b

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the dense connected layer.

        The method takes the derivative of the cost function by the
        output and computes the updates for the weight matrix,
        the bias, and the derivative of the cost function by
        the input.

        Parameters
        ----------
        delta : np.ndarray
            Input array with the derivative of the cost function
            by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """

        # TODO  Implement the backpropagation:
        # 1. finish the computation of delta,
        # 2. compute the weight matrix updates,
        # 3. compute the bias update
        delta = self.afun.backprop(delta)
        self.dW = self.__a.T @ delta
        self.db = np.sum(delta, axis=0)
        # TODO Return modified delta
        return delta @ self.W.T

    def get_output_shape(self) -> tuple[int]:
        """
        Return the shape of the outputs.

        Returns
        -------
        tuple[int]
            Output shape.
        """
        return (self.no,)

    def update(self) -> None:
        """
        Update the weight matrix and the bias.

        The weight matrix and the bias get updated
        with the optimizer set on initialization and
        the updates computed by the backpropagation method.
        """

        # TODO Update the weights and bias with the SGD method
        # Note that the learning rate is stored in the attribute
        # learning_rate.
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db