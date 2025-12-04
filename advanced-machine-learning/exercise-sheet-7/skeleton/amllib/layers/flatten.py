"Implementation of a flattening layer"

import numpy as np
from typing import Optional

from .layer import Layer

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.layers.flatten"
__package__ = "amllib.layers"


class Flatten(Layer):
    """
    Class representation of a flattening layer.

    This layer is responsible for flattening 4-tensors
    (NCHW format) to matrices. The first dimension is kept,
    the others are flattened.

    Attributes
    ----------
    input_shape: tuple[int, int, int]
        Shape of the inputs (without the batch dimension)
    output_shape: int
        Number of flattened outputs per batch.
    """

    def __init__(self, input_shape: Optional[tuple[int, int, int, int]] = None) -> None:
        """
        Initialize the flattening layer.

        The flattening layer is initialized and
        the output shape is computed from input_shape.

        Parameters
        ----------
        input_shape : tuple[int, int, int, int]
            Shape of the inputs (without the batch dimension).
        """
        super().__init__()

        if input_shape is not None:
            self.build(input_shape)

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Build the flattening layer.

        Sets the input shape and the number of outputs
        of the layer.

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            _description_
        """
        self.input_shape = input_shape
        c, h, w = input_shape
        self.output_shape = c * h * w

        self.built = True

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Flatten the input.

        Flatten a tensor of shape (n, c, h, w) to a matrix.
        The first dimension is kept, and the rest is flattened.

        Parameters
        ----------
        input : np.ndarray
            Input tensor of shape (n, c, h, w).

        Returns
        -------
        np.ndarray
            Output matrix of shape (n, c * h * w).

        Raises
        ------
        ValueError
            Raised if the input shape does not match the input shape of the layer.
        """
        n, c, h, w = input.shape

        # Build layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build((c, h, w))

        if (c, h, w) != self.input_shape:
            raise ValueError(f"Input is supposed to have shape {self.input_shape},"
                             f"but it has shape {(c, h, w)}!")

        return input.reshape(n, self.output_shape)

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Flatten the input.

        Flatten a tensor of shape (n, c, h, w) to a matrix.
        The first dimension is kept, and the rest is flattened.
        Data is cached for backpropagation.

        Parameters
        ----------
        input : np.ndarray
            Input tensor of shape (n, c, h, w).

        Returns
        -------
        np.ndarray
            Output matrix of shape (n, c * h * w).

        Raises
        ------
        ValueError
            Raised if the input shape does not match the input shape of the layer.
        """
        n, c, h, w = input.shape

        # Build layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build((c, h, w))

        if (c, h, w) != self.input_shape:
            raise ValueError(f"Input is supposed to have shape {self.input_shape},"
                             f"but it has shape {(c, h, w)}!")

        # TODO Implement the feedforward step of the flattening layer here.
        # Reshape the 4D input to a Matrix. Flatten all channels of all input
        # images.

        pass

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the layer.

        The derivative of the cost function by
        the output is reshaped to an 4-tensor.

        Parameters
        ----------
        delta : np.ndarray
            Input for the backpropagation. Derivative of the cost function
            by the output matrix.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the output reshaped to
            an 4-tensor.
        """

        # TODO Implement the backpropagation for a flattening layer.
        # The 2D input has to be reshaped into a 4-Tensor according
        # to the shape of the last input to the feedforward method.

        pass

    def update(self):
        """
        Update the parameters of the layer.

        Since this layer does not have any trainable
        parameters, nothing has to be updated.
        """
        pass

    def get_output_shape(self) -> tuple[int]:

        return self.output_shape
