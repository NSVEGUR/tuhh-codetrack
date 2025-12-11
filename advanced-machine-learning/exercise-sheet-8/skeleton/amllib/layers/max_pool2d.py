"""Implementation of a 2D pooling layer"""

__author__ = "Jens-Peter M. Zemke, Jonas Grams"
__version__ = "1.1"

__name__ = "amllib.layers.pool2d"
__package__ = "amllib.layers"

import numpy as np

from .layer import Layer

class MaxPool2D(Layer):
    """
    Class representation of an 2D max pooling layer.

    This class implements the 2D max pooling layer.
    For previously defined areas the values are reduced
    to the maximal value in these areas. This reduces
    the size of the inputs.

    Attributes
    ----------
    area: tuple[int, int]
        Shape of the pooling areas. On patches of this shape
        the maximum is computed.
    shape: tuple[int, int ,int, int]
        Shape of the last input batch (With the batch dimension).
    mask: np.ndarray
        Mask with ones at the positions of the last computed maxima.
        This is used in the backpropagation process.
    strict : bool
            Flag for the handling of multiple maxima in a pooling batch.
            True if the entres should be scaled by the number of found
            maxima.
    """

    def __init__(self, area: tuple[int, int], strict: bool = False):
        """
        Initialize the pooling layer.

        Parameters
        ----------
        area : tuple[int, int]
            Pooling area shape.
        strict : bool, optional
            Flag for the handling of multiple maxima in a pooling batch.
            True if the entres should be scaled by the number of found
            maxima, by default False.
        """
        super().__init__()

        self.area = area
        self.shape = None
        self.mask = None
        self.strict = strict
        self.input_shape = None

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Build the pooling layer for the given input shape.

        Nothing has to be done here.

        Parameters
        ----------
        input_shape : tuple[int, int , int]
            Shape of the inputs.
        """
        self.built = True
        self.input_shape = input_shape

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the pooling layer.

        The input is devided into batches of shape self.area
        and then the maxima are assembled.

        Parameters
        ----------
        input : np.ndarray
            Input of the pooling layer.

        Returns
        -------
        np.ndarray
            Pooled input.
        """
        n, c, h, w = self.shape = input.shape

        # Build the layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build(input.shape[1:])

        ph, pw = self.area
        nh, nw = h // ph, w // pw
        oh, ow = h % ph, w % pw

        # Reduce a, if hight and width are odd
        a_reduced = input[:, :, :h-oh, :w-ow]

        # Reshape reduced a to find maxima
        a_reshaped = a_reduced.reshape(n, c, nh, ph, nw, pw)

        # Find maxima and create mask
        z = a_reshaped.max(axis=(3, 5))

        return z

    def feedforward(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluate the pooling layer.

        The input is devided into batches of shape self.area
        and then the maxima are assembled. For backpropagation
        a mask with the positions of the maxima is stored.

        Parameters
        ----------
        input : np.ndarray
            Input of the pooling layer.

        Returns
        -------
        np.ndarray
            Pooled input.
        """

        n, c, h, w = self.shape = input.shape

        # Build the layer with the shape of the inputs, if not done yet.
        if not self.built:
            self.build(input.shape[1:])

        ph, pw = self.area
        nh, nw = h // ph, w // pw
        oh, ow = h % ph, w % pw

        # Reduce a if height and width are odd
        a_reduced = input[:, :, :h-oh, :w-ow]

        # Reshape reduced a to find maxima
        a_reshaped = a_reduced.reshape(n, c, nh, ph, nw, pw)

        # Find maxima and create mask
        z = a_reshaped.max(axis=(3, 5))
        z_newaxis = z[:, :, :, np.newaxis, :, np.newaxis]
        self.mask = (a_reshaped == z_newaxis)

        return z

    def backprop(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagate through the pooling layer.

        Based of the derivative of the cost function
        by the output compute the derivative of the
        cost function by the input. Since this layer has
        no learnable parameters, no updates have to be
        computed.

        Parameters
        ----------
        delta : np.ndarray
            Derivative of the cost function by the output.

        Returns
        -------
        np.ndarray
            Derivative of the cost function by the input.
        """

        n, c, h, w = self.shape
        ph, pw = self.area

        # New delta should be of shape (n, c, h, w).
        # It can be recovered from the reshaped form
        da_reshaped = np.zeros((n, c, h//ph, ph, w//pw, pw))

        # Broadcast delta to shape of da_reshaped
        delta_newaxis = delta[:, :, :, np.newaxis, :, np.newaxis]
        delta_broadcast, _ = np.broadcast_arrays(delta_newaxis, da_reshaped)

        # Set points, where maximums where found to 1
        da_reshaped[self.mask] = delta_broadcast[self.mask]

        # Strict mode for correction of the subgradients in case of multiple maxima
        if self.strict:
            da_reshaped /= np.sum(self.mask, axis=(3, 5), keepdims=True)

        # Build da from da_reshaped
        da = np.zeros(self.shape)
        dah, daw = h - h % ph, w - w % pw
        da[:, :, :dah, :daw] = da_reshaped.reshape(n, c, dah, daw)

        return da

    def update(self):
        """
        Update method for the pooling layer.
        Since it has no learnable parameters,
        nothing has to be done.
        """

        pass

    def get_output_shape(self) -> tuple[int, int, int]:

        c, h, w = self.input_shape
        ph, pw = self.area

        return c, h // ph, w // pw
